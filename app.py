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
from transformers import pipeline
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.plot import plot_plotly
import feedparser

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

# --- Sentiment Analysis Model ---
@st.cache_resource
def load_sentiment_models():
    """한/영 감성 분석 모델을 모두 로드합니다."""
    with st.spinner("한/영 감성 분석 AI 모델을 로드하는 중..."):
        models = {
            'ko': pipeline("sentiment-analysis", model="snunlp/KR-FinBERT-SC"),
            'en': pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        }
    return models

# --- Data Fetching & Processing Functions ---
@st.cache_data(ttl=3600)
def get_categories_from_bq(_client):
    """BigQuery에서 고유 카테고리 목록만 빠르게 가져옵니다."""
    project_id = _client.project
    table_id = f"{project_id}.data_explorer.tds_data"
    try:
        with st.spinner("BigQuery에서 카테고리 목록을 불러오는 중..."):
            query = f"SELECT DISTINCT Category FROM `{table_id}` WHERE Category IS NOT NULL ORDER BY Category"
            df = _client.query(query).to_dataframe()
        return sorted(df['Category'].astype(str).unique())
    except Exception as e:
        print(f"Could not fetch categories (table might not exist yet): {e}")
        return []

def get_trade_data_from_bq(client, categories):
    """BigQuery의 tds_data 테이블에서 선택된 카테고리의 데이터만 로드합니다."""
    if not categories: return pd.DataFrame()
    project_id = client.project; table_id = f"{project_id}.data_explorer.tds_data"
    try:
        query_params = [bigquery.ArrayQueryParameter("categories", "STRING", categories)]
        sql = f"SELECT * FROM `{table_id}` WHERE Category IN UNNEST(@categories)"
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        with st.spinner(f"BigQuery에서 선택된 {len(categories)}개 카테고리 데이터를 로드하는 중..."):
            df = client.query(sql, job_config=job_config).to_dataframe()
        for col in df.columns:
            if 'price' in col.lower() or 'value' in col.lower() or 'volume' in col.lower():
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"BigQuery에서 TDS 데이터를 읽는 중 오류 발생: {e}"); return pd.DataFrame()

def deduplicate_and_write_to_bq(client, df_new, table_name):
    """BigQuery 테이블에 중복을 제거하여 데이터를 씁니다."""
    project_id = client.project
    table_id = f"{project_id}.data_explorer.{table_name}"
    try:
        try:
            sql = f"SELECT * FROM `{table_id}`"
            df_existing = client.query(sql).to_dataframe()
        except Exception:
            df_existing = pd.DataFrame()

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        subset_cols = ['Title', 'Keyword', 'Language'] if 'Title' in df_combined.columns else None
        df_deduplicated = df_combined.drop_duplicates(subset=subset_cols)

        with st.spinner(f"중복을 제거한 데이터를 BigQuery '{table_name}' 테이블에 저장하는 중..."):
            pandas_gbq.to_gbq(df_deduplicated, table_id, project_id=project_id, if_exists="replace", credentials=client._credentials)
        st.sidebar.success(f"데이터가 BigQuery '{table_name}'에 업데이트되었습니다.")
    except Exception as e:
        st.error(f"BigQuery 저장 중 오류 발생: {e}")

def add_trade_data_to_bq(client, df):
    """새로운 수출입 데이터를 BigQuery 테이블에 중복 없이 추가합니다."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
    deduplicate_and_write_to_bq(client, df, "tds_data")

def fetch_historical_news(client, keywords, start_date, end_date, models):
    """[1단계: 일회성] newspaper3k로 과거 데이터를 최대한 수집합니다."""
    all_news_data = []
    for keyword in keywords:
        for lang, country in [('ko', 'KR'), ('en', 'US')]:
            with st.spinner(f"과거 뉴스 수집 중: '{keyword}' ({lang})... (시간이 매우 오래 걸릴 수 있습니다)"):
                news_url = f"https://news.google.com/search?q={keyword}&hl={lang}&gl={country}&ceid={country}%3A{lang}"
                paper = build(news_url, memoize_articles=False, language=lang)
                keyword_articles = []
                for article in paper.articles[:50]: # Limit articles to avoid timeout
                    try:
                        article.download(); article.parse()
                        pub_date = article.publish_date
                        if pub_date and start_date <= pub_date.replace(tzinfo=None) <= end_date:
                            model = models[lang]
                            analysis = model(article.title[:256])[0]
                            score = analysis['score'] if analysis['label'].lower() in ['positive', '5 stars'] else -analysis['score']
                            keyword_articles.append({'Date': pub_date.date(), 'Title': article.title, 'Sentiment': score, 'Keyword': keyword, 'Language': lang})
                    except Exception: continue
                if keyword_articles: all_news_data.append(pd.DataFrame(keyword_articles))
    
    if not all_news_data: st.sidebar.warning("수집된 과거 뉴스가 없습니다."); return
    final_df = pd.concat(all_news_data, ignore_index=True)
    deduplicate_and_write_to_bq(client, final_df, "news_sentiment_cache")
    return final_df

def fetch_latest_news_rss(client, keywords, models):
    """[2단계: 지속적] RSS 피드로 최신 뉴스를 안정적으로 수집합니다."""
    all_news_data = []
    for keyword in keywords:
        for lang, country in [('ko', 'KR'), ('en', 'US')]:
            with st.spinner(f"최신 뉴스 수집 중: '{keyword}' ({lang})..."):
                rss_url = f"https://news.google.com/rss/search?q={keyword}&hl={lang}&gl={country}&ceid={country}:{lang}"
                feed = feedparser.parse(rss_url)
                keyword_articles = []
                for entry in feed.entries[:25]:
                    try:
                        article = Article(entry.link)
                        article.download(); article.parse()
                        pub_date = article.publish_date if article.publish_date else datetime.now()
                        model = models[lang]
                        analysis = model(article.title[:256])[0]
                        score = analysis['score'] if analysis['label'].lower() in ['positive', '5 stars'] else -analysis['score']
                        keyword_articles.append({'Date': pub_date.date(), 'Title': article.title, 'Sentiment': score, 'Keyword': keyword, 'Language': lang})
                    except Exception: continue
                if keyword_articles: all_news_data.append(pd.DataFrame(keyword_articles))

    if not all_news_data: st.sidebar.warning("수집된 최신 뉴스가 없습니다."); return
    final_df = pd.concat(all_news_data, ignore_index=True)
    deduplicate_and_write_to_bq(client, final_df, "news_sentiment_cache")
    return final_df

def fetch_yfinance_data(tickers, start_date, end_date):
    all_data = []
    for name, ticker in tickers.items():
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty:
            df = data[['Close']].copy().reset_index().rename(columns={'Date': '조사일자', 'Close': f'{name}_선물가격_USD'})
            all_data.append(df)
    if not all_data: return pd.DataFrame()
    return reduce(lambda left, right: pd.merge(left, right, on='조사일자', how='outer'), all_data)

def fetch_trends_data(keywords, start_date, end_date, naver_keys):
    all_data = []
    for keyword in keywords:
        keyword_dfs = []
        with st.spinner(f"'{keyword}' 검색량 데이터를 API에서 가져오는 중..."):
            pytrends = TrendReq(hl='ko-KR', tz=540)
            timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='KR', gprop='')
            google_df = pytrends.interest_over_time()
            if not google_df.empty and keyword in google_df.columns:
                google_df_renamed = google_df.reset_index().rename(columns={'date': '날짜', keyword: f'Google_{keyword}'})
                keyword_dfs.append(google_df_renamed)

            if naver_keys['id'] and naver_keys['secret']:
                url = "https://openapi.naver.com/v1/datalab/search"
                body = json.dumps({"startDate": start_date.strftime('%Y-%m-%d'), "endDate": end_date.strftime('%Y-%m-%d'), "timeUnit": "date", "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]})
                request = urllib.request.Request(url)
                request.add_header("X-Naver-Client-Id", naver_keys['id']); request.add_header("X-Naver-Client-Secret", naver_keys['secret']); request.add_header("Content-Type", "application/json")
                response = urllib.request.urlopen(request, data=body.encode("utf-8"))
                if response.getcode() == 200:
                    naver_raw = json.loads(response.read().decode('utf-8'))
                    naver_df = pd.DataFrame(naver_raw['results'][0]['data']).rename(columns={'period': '날짜', 'ratio': f'Naver_{keyword}'})
                    naver_df['날짜'] = pd.to_datetime(naver_df['날짜'])
                    keyword_dfs.append(naver_df)
            
            if keyword_dfs:
                all_data.append(reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), keyword_dfs))

    if not all_data: return pd.DataFrame()
    return reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_data)

def fetch_kamis_data(item_info, start_date, end_date, kamis_keys):
    all_data = []
    date_range = pd.date_range(start=start_date, end=end_date)
    if len(date_range) > 180: st.sidebar.warning(f"KAMIS 조회 기간이 {len(date_range)}일로 깁니다. 오래 걸릴 수 있습니다.")
    progress_bar = st.sidebar.progress(0, text="KAMIS 데이터 조회 중...")
    
    for i, date in enumerate(date_range):
        date_str = date.strftime('%Y-%m-%d')
        url = (f"http://www.kamis.or.kr/service/price/xml.do?p_product_cls_code=02&p_item_category_code={item_info['cat_code']}"
               f"&p_item_code={item_info['item_code']}&p_regday={date_str}&p_convert_kg_yn=Y"
               f"&p_cert_key={kamis_keys['key']}&p_cert_id={kamis_keys['id']}&p_returntype=json")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                items = data.get("data", {}).get("item", [])
                if items:
                    price_str = items[0].get('dpr1', '0').replace(',', '')
                    if price_str.isdigit() and int(price_str) > 0:
                        all_data.append({'조사일자': date, '도매가격_원': int(price_str)})
        except Exception: continue
        finally: progress_bar.progress((i + 1) / len(date_range), text=f"KAMIS 데이터 조회 중... {date_str}")
    
    progress_bar.empty()
    if not all_data: st.sidebar.warning("해당 기간에 대한 KAMIS 데이터가 없습니다."); return pd.DataFrame()
    df = pd.DataFrame(all_data)
    return df

# --- Constants & App ---
st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 분석 대시보드")

bq_client = get_bq_connection()
sentiment_models = load_sentiment_models()
if bq_client is None: st.stop()

st.sidebar.header("⚙️ 분석 설정")

# --- App Startup Workflow ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'categories' not in st.session_state:
    st.session_state.categories = get_categories_from_bq(bq_client)

st.sidebar.subheader("1. 분석 대상 설정")
if not st.session_state.categories:
    st.sidebar.warning("BigQuery에 데이터가 없습니다. 아래에서 새 데이터를 추가해주세요.")
else:
    selected_categories = st.sidebar.multiselect("분석할 품목 카테고리 선택", st.session_state.categories)
    if st.sidebar.button("🚀 선택 완료 및 분석 시작", disabled=(not st.session_state.categories)):
        if not selected_categories:
            st.sidebar.warning("카테고리를 선택해주세요.")
        else:
            st.session_state.raw_trade_df = get_trade_data_from_bq(bq_client, selected_categories)
            if st.session_state.raw_trade_df is not None and not st.session_state.raw_trade_df.empty:
                st.session_state.data_loaded = True
                st.session_state.selected_categories = selected_categories
                st.rerun()
            else:
                st.sidebar.error("데이터를 불러오지 못했습니다.")

with st.sidebar.expander("➕ 새 수출입 데이터 추가"):
    uploaded_file = st.file_uploader("새 파일 업로드하여 BigQuery에 추가", type=['csv', 'xlsx'])
    if uploaded_file:
        if st.button("업로드 파일 BigQuery에 저장"):
            df_new = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            add_trade_data_to_bq(bq_client, df_new)
            st.session_state.clear()
            st.rerun()

if not st.session_state.data_loaded:
    st.info("👈 사이드바에서 분석할 카테고리를 선택하고 '분석 시작' 버튼을 눌러주세요.")
    st.stop()

# --- Analysis UI ---
raw_trade_df = st.session_state.raw_trade_df
selected_categories = st.session_state.selected_categories
st.sidebar.success(f"**{', '.join(selected_categories)}** 데이터 로드 완료!")
st.sidebar.markdown("---")

try:
    file_start_date, file_end_date = raw_trade_df['Date'].min(), raw_trade_df['Date'].max()
    default_keywords = ", ".join(selected_categories) if selected_categories else ""
    keyword_input = st.sidebar.text_input("검색어/뉴스 분석 키워드 입력", default_keywords)
    search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
    st.sidebar.subheader("분석 기간 설정")
    start_date_input = st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date)
    end_date_input = st.sidebar.date_input('종료일', file_end_date, min_value=start_date_input, max_value=file_end_date)
    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)
except Exception as e:
    st.error(f"데이터 처리 중 오류: {e}"); st.stop()

# --- External Data Loading Section ---
COFFEE_TICKERS_YFINANCE = {"미국 커피 C": "KC=F", "런던 로부스타": "RC=F"}
KAMIS_CATEGORIES = {"채소류": "100", "과일류": "200", "축산물": "300", "수산물": "400"}
KAMIS_ITEMS = {"채소류": {"배추": "111", "무": "112", "양파": "114", "마늘": "141"}, "과일류": {"사과": "211", "바나나": "214", "아보카도": "215"}, "축산물": {"소고기": "311", "돼지고기": "312"}, "수산물": {"고등어": "411", "오징어": "413"}}

st.sidebar.subheader("🔗 외부 가격 데이터")
is_coffee_selected = any('커피' in str(cat) for cat in selected_categories)
if is_coffee_selected:
    st.sidebar.info("Yahoo Finance에서 선물가격을 가져옵니다.")
    if st.sidebar.button("선물가격 데이터 가져오기"):
        df = fetch_yfinance_data(COFFEE_TICKERS_YFINANCE, start_date, end_date)
        st.session_state['wholesale_data'] = df
else:
    st.sidebar.info("KAMIS에서 농산물 도매가격 데이터를 가져옵니다.")
    kamis_api_key = st.sidebar.text_input("KAMIS API Key", type="password")
    kamis_api_id = st.sidebar.text_input("KAMIS API ID", type="password")
    cat_name = st.sidebar.selectbox("품목 분류 선택", list(KAMIS_CATEGORIES.keys()))
    if cat_name:
        item_name = st.sidebar.selectbox("세부 품목 선택", list(KAMIS_ITEMS[cat_name].keys()))
        if st.sidebar.button("KAMIS 데이터 가져오기"):
            if kamis_api_key and kamis_api_id:
                item_info = {'item_code': KAMIS_ITEMS[cat_name][item_name], 'cat_code': KAMIS_CATEGORIES[cat_name]}
                df = fetch_kamis_data(item_info, start_date, end_date, {'key': kamis_api_key, 'id': kamis_api_id})
                st.session_state['wholesale_data'] = df
            else: st.sidebar.error("KAMIS API Key와 ID를 모두 입력해주세요.")
raw_wholesale_df = st.session_state.get('wholesale_data', pd.DataFrame())

# --- Search Data Loading Section ---
st.sidebar.subheader("📰 검색량 데이터")
naver_client_id = st.sidebar.text_input("Naver API Client ID", type="password")
naver_client_secret = st.sidebar.text_input("Naver API Client Secret", type="password")
if st.sidebar.button("검색량 데이터 가져오기"):
    if not search_keywords: st.sidebar.warning("검색어를 먼저 입력해주세요.")
    else:
        df = fetch_trends_data(search_keywords, start_date, end_date, {'id': naver_client_id, 'secret': naver_client_secret})
        st.session_state['search_data'] = df
raw_search_df = st.session_state.get('search_data', pd.DataFrame())

# --- News Analysis Section ---
st.sidebar.subheader("📰 뉴스 감성 분석")
if st.sidebar.button("최신 뉴스 분석하기 (RSS)"):
    if not search_keywords: st.sidebar.warning("분석할 키워드를 먼저 입력해주세요.")
    else:
        df = fetch_latest_news_rss(bq_client, search_keywords, sentiment_models)
        st.session_state['news_data'] = df
        st.rerun()

with st.sidebar.expander("⏳ 과거 뉴스 데이터 일괄 수집 (일회성, 매우 느림)"):
    st.warning("이 기능은 지난 1년간의 뉴스를 수집하며, 몇십 분 이상 소요될 수 있고 불안정할 수 있습니다. 일회성으로만 사용하세요.")
    if st.button("과거 뉴스 수집 시작"):
        one_year_ago = datetime.now() - timedelta(days=365)
        df = fetch_historical_news(bq_client, search_keywords, one_year_ago, datetime.now(), sentiment_models)
        st.session_state['news_data'] = df
        st.rerun()
raw_news_df = st.session_state.get('news_data', pd.DataFrame())

# --- Main Display Area ---
tab_list = ["1️⃣ 원본 데이터", "2️⃣ 데이터 표준화", "3️⃣ 뉴스 감성 분석", "4️⃣ 상관관계 분석", "📈 시계열 분해 및 예측"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

with tab1:
    st.subheader("A. 수출입 데이터"); st.dataframe(raw_trade_df.head())
    st.subheader("B. 외부 가격 데이터"); st.dataframe(raw_wholesale_df.head())
    st.subheader("C. 검색량 데이터"); st.dataframe(raw_search_df.head())
    st.subheader("D. 뉴스 데이터"); st.dataframe(raw_news_df.head())

with tab2:
    st.header("데이터 표준화: 같은 기준으로 데이터 맞춰주기")
    trade_df_in_range = raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)]
    filtered_trade_df = trade_df_in_range[trade_df_in_range['Category'].isin(selected_categories)].copy()
    
    if not filtered_trade_df.empty:
        filtered_trade_df.set_index('Date', inplace=True)
        trade_weekly = filtered_trade_df.resample('W-Mon').agg(수입액_USD=('Value', 'sum'), 수입량_KG=('Volume', 'sum')).copy()
        trade_weekly['수입단가_USD_KG'] = trade_weekly['수입액_USD'] / trade_weekly['수입량_KG']
        
        wholesale_weekly = pd.DataFrame()
        if not raw_wholesale_df.empty:
            raw_wholesale_df['조사일자'] = pd.to_datetime(raw_wholesale_df['조사일자'])
            wholesale_weekly = raw_wholesale_df.set_index('조사일자').resample('W-Mon').mean(numeric_only=True)
            if '도매가격_원' in wholesale_weekly.columns:
                wholesale_weekly['도매가격_USD'] = wholesale_weekly['도매가격_원'] / 1350
                wholesale_weekly.drop(columns=['도매가격_원'], inplace=True)

        search_weekly = pd.DataFrame()
        if not raw_search_df.empty:
            raw_search_df['날짜'] = pd.to_datetime(raw_search_df['날짜'])
            search_weekly = raw_search_df.set_index('날짜').resample('W-Mon').mean(numeric_only=True)

        news_weekly = pd.DataFrame()
        if not raw_news_df.empty:
            news_df_in_range = raw_news_df[(raw_news_df['Date'] >= start_date) & (raw_news_df['Date'] <= end_date)]
            if not news_df_in_range.empty:
                news_weekly = news_df_in_range.set_index('Date').resample('W-Mon').agg(뉴스감성점수=('Sentiment', 'mean')).copy()
        
        st.session_state['trade_weekly'] = trade_weekly
        st.session_state['wholesale_weekly'] = wholesale_weekly
        st.session_state['search_weekly'] = search_weekly
        st.session_state['news_weekly'] = news_weekly
        st.write("### 주별 집계 데이터 샘플"); st.dataframe(trade_weekly.head())
        st.dataframe(wholesale_weekly.head()); st.dataframe(search_weekly.head()); st.dataframe(news_weekly.head())
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
        st.session_state['final_df'] = final_df # Save for next tab
        
        st.subheader("통합 데이터 시각화")
        if not final_df.empty:
            df_to_plot = final_df.reset_index().rename(columns={'index': '날짜'})
            df_long = df_to_plot.melt(id_vars='날짜', var_name='데이터 종류', value_name='값')
            fig = px.line(df_long, x='날짜', y='값', color='데이터 종류', 
                          labels={'값': '값', '날짜': '날짜'}, title="최종 통합 데이터 시계열 추이")
            st.plotly_chart(fig, use_container_width=True)

        if len(final_df.columns) > 1:
            st.markdown("---"); st.subheader("상관관계 분석")
            st.write("#### 상관관계 히트맵")
            corr_matrix = final_df.corr(numeric_only=True)
            fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1])
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.write("#### 시차별 상관관계 분석")
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
                    st.info(f"""- **양수 Lag (+)**: **'{col2_name}'** (원인)이 '{col1_name}'(결과)보다 **나중에** 움직일 때의 상관관계입니다. \n- **음수 Lag (-)**: **'{col2_name}'** (원인)이 '{col1_name}'(결과)보다 **먼저** 움직일 때의 상관관계를 의미합니다.""")
            else: st.warning("상관관계를 비교하려면 '수입' 관련 변수와 '외부' 변수가 모두 필요합니다.")
    else: 
        st.warning("2단계에서 처리된 데이터가 없습니다.")

with tab5:
    st.header("시계열 분해 및 예측")
    final_df = st.session_state.get('final_df', pd.DataFrame())
    if not final_df.empty:
        forecast_col = st.selectbox("예측 대상 변수 선택", final_df.columns)
        
        if forecast_col:
            ts_data = final_df[[forecast_col]].dropna()
            if len(ts_data) < 104: # Period is 52, so need at least 2 years of data
                st.warning(f"시계열 분해 및 예측을 위해서는 최소 2년(104주) 이상의 데이터가 필요합니다. 현재 데이터는 {len(ts_data)}주입니다.")
            else:
                st.subheader(f"'{forecast_col}' 시계열 분해")
                decomposition = seasonal_decompose(ts_data[forecast_col], model='additive', period=52)
                
                fig_decompose = go.Figure()
                fig_decompose.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'))
                fig_decompose.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                fig_decompose.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                st.plotly_chart(fig_decompose, use_container_width=True)
                st.write("#### 불규칙 요소 (Residual)")
                st.line_chart(decomposition.resid)


                st.subheader(f"'{forecast_col}' 미래 12주 예측 (by Prophet)")
                prophet_df = ts_data.reset_index().rename(columns={'index': 'ds', forecast_col: 'y'})
                
                m = Prophet()
                m.fit(prophet_df)
                
                future = m.make_future_dataframe(periods=12, freq='W')
                forecast = m.predict(future)

                fig_forecast = plot_plotly(m, forecast)
                fig_forecast.update_layout(title=f"'{forecast_col}' 미래 예측 결과", xaxis_title='날짜', yaxis_title='예측값')
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.write("#### 예측 데이터 테이블")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
    else:
        st.info("4번 탭에서 데이터가 성공적으로 통합되어야 예측을 수행할 수 있습니다.")

