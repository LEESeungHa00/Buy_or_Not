import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from functools import reduce
from pytrends.request import TrendReq
import json
import urllib.request
import yfinance as yf
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas_gbq
from newspaper import build, Article
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.plot import plot_plotly
import feedparser
from urllib.parse import quote
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase

# --- BigQuery Connection (Manual Method for Stability) ---
@st.cache_resource
def get_bq_connection():
    """BigQuery에 직접 연결하고 클라이언트 객체를 반환합니다."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        client = bigquery.Client(credentials=creds, project=creds.project_id)
        return client
    except Exception as e:
        st.error(f"Google BigQuery 연결에 실패했습니다. secrets.toml의 [gcp_service_account] 설정을 확인하세요: {e}")
        return None

# --- Sentiment Analysis Model & Explainer ---
@st.cache_resource
def load_sentiment_assets():
    """한/영 감성 분석 모델 및 설명 도구를 로드합니다."""
    with st.spinner("한/영 감성 분석 AI 모델 및 설명 도구를 로드하는 중..."):
        assets = {
            'ko': {
                'model': AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBERT-SC"),
                'tokenizer': AutoTokenizer.from_pretrained("snunlp/KR-FinBERT-SC")
            },
            'en': {
                'model': AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english"),
                'tokenizer': AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            }
        }
    return assets

def explain_sentiment(model, tokenizer, text, lang):
    """LayerIntegratedGradients를 사용하여 단어 기여도를 계산합니다."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
    embeddings = model.base_model.embeddings if hasattr(model.base_model, 'embeddings') else model.distilbert.embeddings
    lig = LayerIntegratedGradients(model, embeddings)
    target_id = model.config.label2id.get('positive', 1) if lang == 'ko' else model.config.label2id.get('POSITIVE', 1)
    attributions, _ = lig.attribute(inputs['input_ids'], reference_inputs=token_reference.generate_reference(len(inputs['input_ids'][0]), device='cpu'), target=target_id, return_convergence_delta=True)
    attributions = attributions.sum(dim=-1).squeeze(0) / torch.norm(attributions)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    word_attributions = [(token.replace('##', ''), attr.item()) for token, attr in zip(tokens, attributions) if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]]
    return json.dumps(word_attributions, ensure_ascii=False)

# --- Data Fetching & Processing Functions ---
@st.cache_data(ttl=3600)
def get_categories_from_bq(_client):
    project_id = _client.project; table_id = f"{project_id}.data_explorer.tds_data"
    try:
        with st.spinner("BigQuery에서 카테고리 목록을 불러오는 중..."):
            query = f"SELECT DISTINCT Category FROM {table_id} WHERE Category IS NOT NULL ORDER BY Category"
            df = _client.query(query).to_dataframe()
        return sorted(df['Category'].astype(str).unique())
    except Exception as e:
        st.error(f"BigQuery 테이블({table_id})을 읽는 중 오류가 발생했습니다. 데이터세트/테이블 이름, 서비스 계정 권한을 확인해주세요. 원본 오류: {e}")
        return []

def get_trade_data_from_bq(client, categories):
    if not categories: return pd.DataFrame()
    project_id = client.project; table_id = f"{project_id}.data_explorer.tds_data"
    try:
        query_params = [bigquery.ArrayQueryParameter("categories", "STRING", categories)]
        sql = f"SELECT * FROM {table_id} WHERE Category IN UNNEST(@categories)"
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        with st.spinner(f"BigQuery에서 {len(categories)}개 카테고리 데이터를 로드하는 중..."):
            df = client.query(sql, job_config=job_config).to_dataframe()
        
        # [수정] 데이터 타입 변환을 더 명시적으로 처리
        for col in df.columns:
            if 'price' in col.lower() or 'value' in col.lower() or 'volume' in col.lower():
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # [수정] 'Date' 컬럼이 최우선, 없다면 다른 날짜 컬럼을 찾아 변환
        date_col_found = None
        if 'Date' in df.columns:
            date_col_found = 'Date'
        elif 'date' in df.columns:
            date_col_found = 'date'
        
        if date_col_found:
             df[date_col_found] = pd.to_datetime(df[date_col_found], errors='coerce')
             df.rename(columns={date_col_found: 'Date'}, inplace=True) # 컬럼명을 'Date'로 통일
        
        return df
    except Exception as e:
        st.error(f"BigQuery에서 TDS 데이터를 읽는 중 오류 발생: {e}"); return pd.DataFrame()

def deduplicate_and_write_to_bq(client, df_new, table_name, subset_cols=None):
    project_id = client.project; table_id = f"{project_id}.data_explorer.{table_name}"
    try:
        try:
            sql = f"SELECT * FROM {table_id}"
            df_existing = client.query(sql).to_dataframe()
        except Exception: df_existing = pd.DataFrame()
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        if subset_cols:
            df_deduplicated = df_combined.drop_duplicates(subset=subset_cols)
        else:
            df_deduplicated = df_combined.drop_duplicates()
        with st.spinner(f"중복을 제거한 데이터를 BigQuery '{table_name}'에 저장하는 중..."):
            pandas_gbq.to_gbq(df_deduplicated, table_id, project_id=project_id, if_exists="replace", credentials=client._credentials)
        st.sidebar.success(f"데이터가 BigQuery '{table_name}'에 업데이트되었습니다.")
    except Exception as e: st.error(f"BigQuery 저장 중 오류 발생: {e}")

def add_trade_data_to_bq(client, df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
    deduplicate_and_write_to_bq(client, df, "tds_data")

def fetch_and_analyze_news_base(client, keywords, start_date, end_date, models, article_source_func):
    project_id = client.project; table_name = "news_sentiment_cache"
    all_news_data = []
    for keyword in keywords:
        for lang, country in [('ko', 'KR'), ('en', 'US')]:
            with st.spinner(f"'{keyword}' ({lang}) 뉴스 수집 및 분석 중..."):
                articles_to_process = article_source_func(keyword, lang, country)
                keyword_articles = []
                model, tokenizer = models[lang]['model'], models[lang]['tokenizer']
                for article in articles_to_process:
                    try:
                        article.download(); article.parse()
                        pub_date = article.publish_date
                        if pub_date and start_date <= pd.to_datetime(pub_date).tz_localize(None) <= end_date:
                            title = article.title[:256]
                            pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                            analysis = pipe(title)[0]
                            label, score = analysis['label'], analysis['score']
                            sentiment_score = score if label.lower() in ['positive', '5 stars'] else -score if label.lower() in ['negative', '1 star'] else 0.0
                            explanation_json = explain_sentiment(model, tokenizer, title, lang)
                            keyword_articles.append({'Date': pub_date.date(), 'Title': title, 'Sentiment': sentiment_score, 'Keyword': keyword, 'Language': lang, 'Explanation': explanation_json})
                    except Exception: continue
                if keyword_articles: all_news_data.append(pd.DataFrame(keyword_articles))
    if not all_news_data: st.sidebar.warning("수집된 뉴스가 없습니다."); return pd.DataFrame()
    final_df = pd.concat(all_news_data, ignore_index=True)
    deduplicate_and_write_to_bq(client, final_df, "news_sentiment_cache", subset_cols=['Title', 'Keyword', 'Language'])
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    return final_df

def fetch_historical_news(client, keywords, start_date, end_date, models):
    def get_articles(keyword, lang, country):
        keyword_encoded = quote(keyword); news_url = f"https://news.google.com/search?q={keyword_encoded}&hl={lang}&gl={country}&ceid={country}%3A{lang}"
        return build(news_url, memoize_articles=False, language=lang).articles[:50]
    return fetch_and_analyze_news_base(client, keywords, start_date, end_date, models, get_articles)

def fetch_latest_news_rss(client, keywords, models):
    def get_articles(keyword, lang, country):
        keyword_encoded = quote(keyword); rss_url = f"https://news.google.com/rss/search?q={keyword_encoded}&hl={lang}&gl={country}&ceid={country}:{lang}"
        feed = feedparser.parse(rss_url)
        return [Article(entry.link) for entry in feed.entries[:25]]
    return fetch_and_analyze_news_base(client, keywords, datetime.now() - timedelta(days=7), datetime.now(), models, get_articles)

def fetch_yfinance_data(tickers, start_date, end_date):
    all_data = []
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty:
            df = data[['Close']].copy().reset_index().rename(columns={'Date': '날짜', 'Close': f'{name}_선물가격_USD'})
            all_data.append(df)
    if not all_data: return pd.DataFrame()
    # [수정] 날짜 컬럼 이름을 '날짜'로 통일하여 merge
    return reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_data)

def call_naver_api(url, body, naver_keys):
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", naver_keys['id']); request.add_header("X-Naver-Client-Secret", naver_keys['secret']); request.add_header("Content-Type", "application/json")
    response = urllib.request.urlopen(request, data=body.encode("utf-8"))
    if response.getcode() == 200:
        return json.loads(response.read().decode('utf-8'))
    return None

def fetch_trends_data(keywords, start_date, end_date, naver_keys):
    all_data = []
    NAVER_SHOPPING_CAT_MAP = {'커피 생두(Green Bean)': "50004457", '아보카도(열대과일)': "50002194"}
    for keyword in keywords:
        keyword_dfs = []
        with st.spinner(f"'{keyword}' 트렌드 데이터를 API에서 가져오는 중..."):
            pytrends = TrendReq(hl='ko-KR', tz=540)
            pytrends.build_payload([keyword], cat=0, timeframe=f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}", geo='KR')
            google_df = pytrends.interest_over_time()
            if not google_df.empty and keyword in google_df.columns:
                if 'isPartial' in google_df.columns: google_df = google_df.drop(columns=['isPartial'])
                keyword_dfs.append(google_df.reset_index().rename(columns={'date': '날짜', keyword: f'Google_{keyword}'}))
            if naver_keys['id'] and naver_keys['secret']:
                body = json.dumps({"startDate": start_date.strftime('%Y-%m-%d'), "endDate": end_date.strftime('%Y-%m-%d'), "timeUnit": "date", "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]})
                search_res = call_naver_api("https://openapi.naver.com/v1/datalab/search", body, naver_keys)
                if search_res: keyword_dfs.append(pd.DataFrame(search_res['results'][0]['data']).rename(columns={'period': '날짜', 'ratio': f'NaverSearch_{keyword}'}))
                if keyword in NAVER_SHOPPING_CAT_MAP:
                    body_shop = json.dumps({"startDate": start_date.strftime('%Y-%m-%d'), "endDate": end_date.strftime('%Y-%m-%d'), "timeUnit": "date", "category": NAVER_SHOPPING_CAT_MAP[keyword], "keyword": keyword})
                    shop_res = call_naver_api("https://openapi.naver.com/v1/datalab/shopping/categories", body_shop, naver_keys)
                    if shop_res: keyword_dfs.append(pd.DataFrame(shop_res['results'][0]['data']).rename(columns={'period': '날짜', 'ratio': f'NaverShop_{keyword}'}))
            if keyword_dfs:
                for i, df in enumerate(keyword_dfs): keyword_dfs[i]['날짜'] = pd.to_datetime(df['날짜'])
                all_data.append(reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), keyword_dfs))
    if not all_data: return pd.DataFrame()
    return reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_data)

def fetch_kamis_data(client, item_info, start_date, end_date, kamis_keys):
    project_id = client.project; table_name = "kamis_cache"
    table_id = f"{project_id}.data_explorer.{table_name}"; item_code = item_info['item_code']; kind_code = item_info['kind_code']
    try:
        sql = f"SELECT Date AS 날짜, Price AS 도매가격_원 FROM {table_id} WHERE ItemCode = '{item_code}' AND KindCode = '{kind_code}' AND Date >= '{start_date.strftime('%Y-%m-%d')}' AND Date <= '{end_date.strftime('%Y-%m-%d')}'"
        df_cache = client.query(sql).to_dataframe()
        if len(df_cache) >= (end_date - start_date).days * 0.8:
            st.sidebar.info(f"'{item_info['item_name']}-{item_info['kind_name']}' KAMIS 데이터를 BigQuery 캐시에서 로드했습니다.")
            df_cache['날짜'] = pd.to_datetime(df_cache['날짜'])
            return df_cache
    except Exception: pass
    all_data = []
    date_range = pd.date_range(start=start_date, end=end_date)
    progress_bar = st.sidebar.progress(0, text="KAMIS 데이터 API 조회 중...")
    for i, date in enumerate(date_range):
        date_str = date.strftime('%Y-%m-%d')
        url = (f"http://www.kamis.or.kr/service/price/xml.do?p_product_cls_code=01&p_regday={date_str}"
               f"&p_item_category_code={item_info['cat_code']}&p_item_code={item_code}&p_kind_code={kind_code}"
               f"&p_product_rank_code={item_info['rank_code']}&p_convert_kg_yn=Y"
               f"&p_cert_key={kamis_keys['key']}&p_cert_id={kamis_keys['id']}&p_returntype=json")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"] and "item" in data["data"]:
                    price_str = data["data"]["item"][0].get('price', '0').replace(',', '')
                    if price_str.isdigit() and int(price_str) > 0:
                        all_data.append({'Date': date, 'Price': int(price_str), 'ItemCode': item_code, 'KindCode': kind_code})
        except Exception: continue
        finally: progress_bar.progress((i + 1) / len(date_range))
    progress_bar.empty()
    if not all_data: st.sidebar.warning("해당 기간에 대한 KAMIS 데이터가 없습니다."); return pd.DataFrame()
    df_new = pd.DataFrame(all_data)
    deduplicate_and_write_to_bq(client, df_new, table_name, subset_cols=['Date', 'ItemCode', 'KindCode'])
    df_new['Date'] = pd.to_datetime(df_new['Date'])
    # [수정] 날짜 컬럼 이름을 '날짜'로 통일
    return df_new.rename(columns={'Date': '날짜', 'Price': '도매가격_원'})

# --- Constants & App ---
COFFEE_TICKERS_YFINANCE = {"미국 커피 C": "KC=F", "런던 로부스타": "RC=F"}
KAMIS_FULL_DATA = {
    '쌀': {'cat_code': '100', 'item_code': '111', 'kinds': {'20kg': '01', '백미': '02', '현미': '03', '10kg': '10'}},
    '감자': {'cat_code': '100', 'item_code': '152', 'kinds': {'수미(노지)': '01', '수미(시설)': '04'}},
    '배추': {'cat_code': '200', 'item_code': '211', 'kinds': {'봄': '01', '여름(고랭지)': '02', '가을': '03', '월동': '06'}},
    '양파': {'cat_code': '200', 'item_code': '245', 'kinds': {'양파': '00', '햇양파': '02', '수입': '10'}},
    '사과': {'cat_code': '400', 'item_code': '411', 'kinds': {'후지': '05', '쓰가루(아오리)': '06', '홍로': '07'}},
    '바나나': {'cat_code': '400', 'item_code': '418', 'kinds': {'수입': '02'}},
    '아보카도': {'cat_code': '400', 'item_code': '430', 'kinds': {'수입': '00'}},
    '고등어': {'cat_code': '600', 'item_code': '611', 'kinds': {'생선': '01', '냉동': '02', '국산(염장)': '03'}},
}
st.set_page_config(layout="wide"); st.title("📊 데이터 탐색 및 통합 분석 대시보드")
bq_client = get_bq_connection(); sentiment_assets = load_sentiment_assets()
if bq_client is None: st.stop()
st.sidebar.header("⚙️ 분석 설정")

if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'categories' not in st.session_state: st.session_state.categories = get_categories_from_bq(bq_client)

st.sidebar.subheader("1. 분석 대상 설정")
if not st.session_state.categories:
    st.sidebar.warning("BigQuery에 데이터가 없습니다. 아래에서 새 데이터를 추가해주세요.")
else:
    selected_categories = st.sidebar.multiselect("분석할 품목 카테고리 선택", st.session_state.categories)
    if st.sidebar.button("🚀 선택 완료 및 분석 시작", disabled=(not st.session_state.categories)):
        if not selected_categories: st.sidebar.warning("카테고리를 선택해주세요.")
        else:
            st.session_state.raw_trade_df = get_trade_data_from_bq(bq_client, selected_categories)
            if st.session_state.raw_trade_df is not None and not st.session_state.raw_trade_df.empty:
                st.session_state.data_loaded = True
                st.session_state.selected_categories = selected_categories
                # [수정] st.rerun() 제거
            else: st.sidebar.error("데이터를 불러오지 못했습니다.")

with st.sidebar.expander("➕ 새 수출입 데이터 추가"):
    uploaded_file = st.file_uploader("새 파일 업로드", type=['csv', 'xlsx'])
    if uploaded_file and st.button("업로드 파일 BigQuery에 저장"):
        try:
            df_new = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            numeric_cols = ['Value', 'Volume', 'Unit_Price', 'UnitPrice']
            for col in numeric_cols:
                if col in df_new.columns:
                    df_new[col] = df_new[col].astype(str).str.replace(',', '').replace('-', np.nan)
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
            if 'Date' in df_new.columns: df_new['Date'] = pd.to_datetime(df_new['Date'], errors='coerce')
            add_trade_data_to_bq(bq_client, df_new)
            # [수정] st.rerun() 제거 및 세션 상태 초기화
            st.session_state.clear()
            st.success("데이터가 추가되었습니다. 페이지를 새로고침하여 분석을 다시 시작하세요.")

if not st.session_state.data_loaded:
    st.info("👈 사이드바에서 분석할 카테고리를 선택하고 '분석 시작' 버튼을 눌러주세요."); st.stop()

# --- Analysis UI ---
raw_trade_df = st.session_state.raw_trade_df; selected_categories = st.session_state.selected_categories
st.sidebar.success(f"**{', '.join(selected_categories)}** 데이터 로드 완료!"); st.sidebar.markdown("---")
try:
    if 'Date' not in raw_trade_df.columns:
        st.error("불러온 수출입 데이터에 'Date' 컬럼이 없습니다. 데이터를 확인해주세요.")
        st.stop()
    raw_trade_df.dropna(subset=['Date'], inplace=True)
    file_start_date, file_end_date = raw_trade_df['Date'].min(), raw_trade_df['Date'].max()
    default_keywords = ", ".join(selected_categories) if selected_categories else ""
    keyword_input = st.sidebar.text_input("검색어/뉴스 분석 키워드 입력", default_keywords)
    search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
    st.sidebar.subheader("분석 기간 설정")
    start_date_input = st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date)
    end_date_input = st.sidebar.date_input('종료일', file_end_date, min_value=start_date_input, max_value=file_end_date)
    start_date = pd.to_datetime(start_date_input); end_date = pd.to_datetime(end_date_input)
except Exception as e: st.error(f"데이터 기간 설정 중 오류 발생: {e}"); st.stop()

raw_wholesale_df = st.session_state.get('wholesale_data', pd.DataFrame())
raw_search_df = st.session_state.get('search_data', pd.DataFrame())
raw_news_df = st.session_state.get('news_data', pd.DataFrame())

st.sidebar.subheader("🔗 외부 데이터 연동")
is_coffee_selected = any('커피' in str(cat) for cat in selected_categories)
if is_coffee_selected:
    if st.sidebar.button("선물가격 데이터 가져오기"):
        st.session_state.wholesale_data = fetch_yfinance_data(COFFEE_TICKERS_YFINANCE, start_date, end_date)
        # [수정] st.rerun() 제거
else:
    st.sidebar.markdown("##### KAMIS 농산물 가격")
    kamis_api_key = st.sidebar.text_input("KAMIS API Key", type="password")
    kamis_api_id = st.sidebar.text_input("KAMIS API ID", type="password")
    item_name = st.sidebar.selectbox("품목 선택", list(KAMIS_FULL_DATA.keys()))
    if item_name:
        kind_name = st.sidebar.selectbox("품종 선택", list(KAMIS_FULL_DATA[item_name]['kinds'].keys()))
        if st.sidebar.button("KAMIS 데이터 가져오기"):
            if kamis_api_key and kamis_api_id:
                item_info = {
                    'item_name': item_name, 'kind_name': kind_name,
                    'item_code': KAMIS_FULL_DATA[item_name]['item_code'],
                    'kind_code': KAMIS_FULL_DATA[item_name]['kinds'][kind_name],
                    'cat_code': KAMIS_FULL_DATA[item_name]['cat_code'],
                    'rank_code': '01'
                }
                st.session_state.wholesale_data = fetch_kamis_data(bq_client, item_info, start_date, end_date, {'key': kamis_api_key, 'id': kamis_api_id})
                # [수정] st.rerun() 제거
            else: st.sidebar.error("KAMIS API Key와 ID를 모두 입력해주세요.")
st.sidebar.markdown("##### 트렌드 데이터")
naver_client_id = st.sidebar.text_input("Naver API Client ID", type="password")
naver_client_secret = st.sidebar.text_input("Naver API Client Secret", type="password")
if st.sidebar.button("트렌드 데이터 가져오기"):
    if not search_keywords: st.sidebar.warning("검색어를 먼저 입력해주세요.")
    else:
        st.session_state.search_data = fetch_trends_data(search_keywords, start_date, end_date, {'id': naver_client_id, 'secret': naver_client_secret})
        # [수정] st.rerun() 제거
st.sidebar.markdown("##### 뉴스 감성 분석")
if st.sidebar.button("최신 뉴스 분석하기 (RSS)"):
    if not search_keywords: st.sidebar.warning("분석할 키워드를 먼저 입력해주세요.")
    else:
        st.session_state.news_data = fetch_latest_news_rss(bq_client, search_keywords, sentiment_assets)
        # [수정] st.rerun() 제거
with st.sidebar.expander("⏳ 과거 뉴스 데이터 일괄 수집"):
    st.warning("일회성 기능으로, 매우 느릴 수 있습니다.")
    if st.button("과거 뉴스 수집 시작"):
        one_year_ago = datetime.now() - timedelta(days=365)
        st.session_state.news_data = fetch_historical_news(bq_client, search_keywords, one_year_ago, datetime.now(), sentiment_assets)
        # [수정] st.rerun() 제거

# --- Main Display ---
tab_list = ["1️⃣ 원본 데이터", "2️⃣ 데이터 표준화", "3️⃣ 뉴스 감성 분석", "4️⃣ 상관관계 분석", "📈 시계열 분해 및 예측"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

with tab1:
    st.subheader("A. 수출입 데이터"); st.dataframe(raw_trade_df.head())
    st.subheader("B. 외부 가격 데이터"); st.dataframe(raw_wholesale_df.head())
    st.subheader("C. 트렌드 데이터"); st.dataframe(raw_search_df.head())
    st.subheader("D. 뉴스 데이터"); st.dataframe(raw_news_df.head())
    
with tab2:
    st.header("데이터 표준화: 같은 기준으로 데이터 맞춰주기")
    trade_df_in_range = raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)]
    filtered_trade_df = trade_df_in_range[trade_df_in_range['Category'].isin(selected_categories)].copy()
    
    if not filtered_trade_df.empty:
        filtered_trade_df.set_index('Date', inplace=True)
        trade_weekly = filtered_trade_df.resample('W-Mon').agg(수입액_USD=('Value', 'sum'), 수입량_KG=('Volume', 'sum')).copy()
        trade_weekly['수입단가_USD_KG'] = trade_weekly['수입액_USD'] / trade_weekly['수입량_KG']
        trade_weekly.index.name = '날짜' # [추가] 인덱스 이름 통일

        wholesale_weekly = pd.DataFrame()
        if not raw_wholesale_df.empty:
            # [수정] 모든 외부 데이터는 '날짜' 컬럼을 기준으로 처리
            raw_wholesale_df['날짜'] = pd.to_datetime(raw_wholesale_df['날짜'])
            wholesale_weekly = raw_wholesale_df.set_index('날짜').resample('W-Mon').mean(numeric_only=True)
            if '도매가격_원' in wholesale_weekly.columns:
                wholesale_weekly['도매가격_USD'] = wholesale_weekly['도매가격_원'] / 1350 # 환율은 예시
                wholesale_weekly.drop(columns=['도매가격_원'], inplace=True)
            wholesale_weekly.index.name = '날짜' # [추가] 인덱스 이름 통일

        search_weekly = pd.DataFrame()
        if not raw_search_df.empty:
            raw_search_df['날짜'] = pd.to_datetime(raw_search_df['날짜'])
            search_weekly = raw_search_df.set_index('날짜').resample('W-Mon').mean(numeric_only=True)
            search_weekly.index.name = '날짜' # [추가] 인덱스 이름 통일

        news_weekly = pd.DataFrame()
        if not raw_news_df.empty:
            raw_news_df['Date'] = pd.to_datetime(raw_news_df['Date'])
            news_df_in_range = raw_news_df[(raw_news_df['Date'] >= start_date) & (raw_news_df['Date'] <= end_date)]
            if not news_df_in_range.empty:
                news_weekly = news_df_in_range.set_index('Date').resample('W-Mon').agg(뉴스감성점수=('Sentiment', 'mean')).copy()
                news_weekly.index.name = '날짜' # [추가] 인덱스 이름 통일
        
        st.session_state['trade_weekly'] = trade_weekly
        st.session_state['wholesale_weekly'] = wholesale_weekly
        st.session_state['search_weekly'] = search_weekly
        st.session_state['news_weekly'] = news_weekly
        st.write("### 주별 집계 데이터 샘플"); 
        st.dataframe(trade_weekly.head())
        st.dataframe(wholesale_weekly.head())
        st.dataframe(search_weekly.head())
        st.dataframe(news_weekly.head())
    else: st.warning("선택된 기간에 해당하는 수출입 데이터가 없습니다.")

with tab3:
    st.header("뉴스 감성 분석 결과")
    if not raw_news_df.empty:
        st.subheader("주별 평균 감성 점수 추이")
        news_weekly_df = st.session_state.get('news_weekly', pd.DataFrame())
        if not news_weekly_df.empty:
            fig = px.line(news_weekly_df, y='뉴스감성점수', title="주별 뉴스 감성 점수")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("전체 뉴스 감성 분포")
            def categorize_sentiment(score):
                if score > 0.1: return "긍정 (Positive)"
                elif score < -0.1: return "부정 (Negative)"
                else: return "중립 (Neutral)"
            raw_news_df['Sentiment_Category'] = raw_news_df['Sentiment'].apply(categorize_sentiment)
            sentiment_counts = raw_news_df['Sentiment_Category'].value_counts().reset_index()
            sentiment_counts.columns = ['감성', '기사 수']
            fig_pie = px.pie(sentiment_counts, names='감성', values='기사 수', title="전체 기사 긍정/부정/중립 비율", color_discrete_map={'긍정 (Positive)':'blue', '부정 (Negative)':'red', '중립 (Neutral)':'grey'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.subheader("키워드별 평균 감성 점수")
            avg_sentiment_by_keyword = raw_news_df.groupby('Keyword')['Sentiment'].mean().reset_index().sort_values(by='Sentiment', ascending=False)
            fig_bar = px.bar(avg_sentiment_by_keyword, x='Keyword', y='Sentiment', title="키워드별 평균 감성 점수 비교", color='Sentiment', color_continuous_scale='RdBu_r', range_color=[-1, 1], labels={'Sentiment': '평균 감성 점수'})
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("---")
        st.subheader("수집된 뉴스 기사 목록 (최신순)"); st.dataframe(raw_news_df.sort_values(by='Date', ascending=False))
    else: st.info("사이드바에서 '뉴스 기사 분석하기' 버튼을 눌러주세요.")

with tab4:
    st.header("상관관계 분석")
    trade_weekly = st.session_state.get('trade_weekly', pd.DataFrame())
    wholesale_weekly = st.session_state.get('wholesale_weekly', pd.DataFrame())
    search_weekly = st.session_state.get('search_weekly', pd.DataFrame())
    news_weekly = st.session_state.get('news_weekly', pd.DataFrame())
    dfs_to_concat = [df for df in [trade_weekly, wholesale_weekly, search_weekly, news_weekly] if not df.empty]
    if dfs_to_concat:
        final_df = pd.concat(dfs_to_concat, axis=1).interpolate(method='linear', limit_direction='forward').dropna(how='all')
        st.session_state['final_df'] = final_df
        st.subheader("통합 데이터 시각화")
        if not final_df.empty:
            # [수정] 인덱스 이름이 '날짜'로 통일되었으므로 reset_index() 후 바로 사용 가능
            df_to_plot = final_df.reset_index()
            df_long = df_to_plot.melt(id_vars='날짜', var_name='데이터 종류', value_name='값')
            fig = px.line(df_long, x='날짜', y='값', color='데이터 종류', labels={'값': '값', '날짜': '날짜'}, title="최종 통합 데이터 시계열 추이")
            st.plotly_chart(fig, use_container_width=True)

        if len(final_df.columns) > 1:
            st.markdown("---"); st.subheader("상관관계 분석")
            st.write("#### 상관관계 히트맵")
            corr_matrix = final_df.corr(numeric_only=True)
            fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1])
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.write("#### ℹ️ 시차별 상관관계 분석")
            base_vars = [col for col in final_df.columns if '수입' in col]
            influencing_vars = [col for col in final_df.columns if '수입' not in col]
            if base_vars and influencing_vars:
                col1_name = st.selectbox("기준 변수 (결과) 선택", base_vars)
                col2_name = st.selectbox("영향 변수 (원인) 선택", influencing_vars)
                @st.cache_data
                def calculate_cross_corr(df, col1, col2, max_lag=12):
                    lags = range(-max_lag, max_lag + 1)
                    correlations = [df[col1].corr(df[col2].shift(lag)) for lag in lags]
                    return pd.DataFrame({'Lag (주)': lags, '상관계수': correlations})
                if col1_name and col2_name:
                    cross_corr_df = calculate_cross_corr(final_df, col1_name, col2_name)
                    fig_cross_corr = px.bar(cross_corr_df, x='Lag (주)', y='상관계수', title=f"'{col1_name}'와 '{col2_name}'의 시차별 상관관계")
                    fig_cross_corr.add_hline(y=0); st.plotly_chart(fig_cross_corr, use_container_width=True)
                    st.info("""**결과 해석 가이드:** ...생략...""")
            else: st.warning("상관관계를 비교하려면 '수입' 관련 변수와 '외부' 변수가 모두 필요합니다.")
    else: st.warning("2단계에서 처리된 데이터가 없습니다.")

with tab5:
    st.header("시계열 분해 및 예측")
    final_df = st.session_state.get('final_df', pd.DataFrame())
    if not final_df.empty:
        forecast_col = st.selectbox("예측 대상 변수 선택", final_df.columns)
        if forecast_col:
            ts_data = final_df[[forecast_col]].dropna()
            # [수정] 분해 및 예측을 위한 데이터 길이 조건 완화 (경고 메시지로 대체)
            if len(ts_data) < 24: # 최소 24주 데이터는 필요
                st.warning(f"시계열 분석을 위해 최소 24주 이상의 데이터가 필요합니다. 현재 데이터는 {len(ts_data)}주입니다.")
            else:
                st.subheader(f"'{forecast_col}' 시계열 분해")
                # [수정] 주별 데이터이므로 period=52로 고정하지 않고 데이터 길이에 맞게 조정
                period = 52 if len(ts_data) >= 104 else int(len(ts_data) / 2)
                decomposition = seasonal_decompose(ts_data[forecast_col], model='additive', period=period)
                fig_decompose = go.Figure()
                fig_decompose.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'))
                fig_decompose.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                fig_decompose.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                st.plotly_chart(fig_decompose, use_container_width=True)
                st.write("#### 불규칙 요소 (Residual)"); st.line_chart(decomposition.resid)
                
                st.subheader(f"'{forecast_col}' 미래 12주 예측 (by Prophet)")
                # [수정] 인덱스 이름이 '날짜'로 통일되었으므로 reset_index() 후 rename
                prophet_df = ts_data.reset_index().rename(columns={'날짜': 'ds', forecast_col: 'y'})
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=12, freq='W')
                forecast = m.predict(future)
                fig_forecast = plot_plotly(m, forecast)
                fig_forecast.update_layout(title=f"'{forecast_col}' 미래 예측 결과", xaxis_title='날짜', yaxis_title='예측값')
                st.plotly_chart(fig_forecast, use_container_width=True)
                st.write("#### 예측 데이터 테이블"); st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
    else:
        st.info("4번 탭에서 데이터가 성공적으로 통합되어야 예측을 수행할 수 있습니다.")
