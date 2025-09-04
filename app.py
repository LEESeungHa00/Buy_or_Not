import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from functools import reduce
import json
import urllib.request
import yfinance as yf
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas_gbq
from newspaper import Article, Config
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.plot import plot_plotly
import feedparser
from urllib.parse import quote
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase

# --- BigQuery Connection ---
@st.cache_resource
def get_bq_connection():
    """BigQuery에 직접 연결하고 클라이언트 객체를 반환합니다."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        client = bigquery.Client(credentials=creds, project=creds.project_id)
        return client
    except Exception as e:
        st.error(f"Google BigQuery 연결 실패: secrets.toml 설정을 확인하세요. 오류: {e}")
        return None

# --- Sentiment Models ---
@st.cache_resource
def load_sentiment_assets():
    with st.spinner("한/영 감성 분석 모델 로드 중..."):
        return {
            'ko': {
                'model': AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBERT-SC"),
                'tokenizer': AutoTokenizer.from_pretrained("snunlp/KR-FinBERT-SC", use_fast=True)
            },
            'en': {
                'model': AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert"),
                'tokenizer': AutoTokenizer.from_pretrained("ProsusAI/finbert", use_fast=True)
            }
        }

# --- Helper functions ---
def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def predict_batch(texts, model, tokenizer, device=DEVICE, max_len=128):
    model.eval()
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                    max_length=max_len, return_offsets_mapping=False)
    if device != "cpu":
        enc = {k: v.to(device) for k, v in enc.items()}
        model.to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = _softmax(logits.cpu().numpy())
    id2label = model.config.id2label
    labels = [id2label[i].lower() for i in range(logits.shape[-1])]
    return probs, labels

def compute_margin_and_pred(probs, labels):
    order = np.argsort(probs)[::-1]
    pred_idx = int(order[0])
    pred_prob = float(probs[pred_idx])
    second = float(probs[order[1]]) if len(order) > 1 else 0.0
    margin = pred_prob - second
    return pred_idx, pred_prob, margin

class ForwardWrapper(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def token_attributions_ig(text, model, tokenizer, target_idx,
                          max_len=128, n_steps=IG_N_STEPS):
    model.eval()
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length',
                    max_length=max_len, return_offsets_mapping=True)
    input_ids, attention_mask = enc['input_ids'], enc['attention_mask']
    offsets = enc['offset_mapping'][0].tolist()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baselines = torch.full_like(input_ids, pad_id)
    fw = ForwardWrapper(model)
    lig = LayerIntegratedGradients(fw, model.get_input_embeddings())
    attributions, _ = lig.attribute(inputs=input_ids,
                                    baselines=baselines,
                                    additional_forward_args=(attention_mask,),
                                    target=target_idx,
                                    n_steps=n_steps)
    token_attr = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
    token_attr = token_attr * attention_mask.squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    keep_idx = [i for i, t in enumerate(tokens)
                if t not in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token)]
    words = []
    for i in keep_idx:
        s, e = offsets[i]
        word = text[s:e]
        words.append((word, float(token_attr[i])))
    if MECAB_AVAILABLE:
        merged = {}
        for w, sc in words:
            try: morphs = mecab.nouns(w) or [w]
            except: morphs = [w]
            for m in morphs: merged[m] = merged.get(m, 0.0) + sc
        words = list(merged.items())
    pos = sorted([x for x in words if x[1] > 0], key=lambda x: -x[1])
    neg = sorted([x for x in words if x[1] < 0], key=lambda x: x[1])
    return pos, neg

def classify_with_conditional_ig(texts, model, tokenizer,
                                batch_size=BATCH_SIZE, topk=TOPK_WORDS):
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        probs_batch, labels = predict_batch(chunk, model, tokenizer)
        for j, text in enumerate(chunk):
            probs = probs_batch[j]
            pred_idx, pred_prob, margin = compute_margin_and_pred(probs, labels)
            pred_label = labels[pred_idx]
            run_ig = (pred_prob < IG_PROB_THRESH) or (margin < IG_MARGIN_THRESH)
            pos_words, neg_words = [], []
            if run_ig:
                try:
                    pos_words, neg_words = token_attributions_ig(text, model, tokenizer, pred_idx)
                except Exception as e:
                    logging.warning(f"IG 실패: {e}")
            top_pos = [w for w,_ in pos_words[:topk]]
            top_neg = [w for w,_ in neg_words[:topk]]
            results.append({
                "text": text,
                "label": pred_label,
                "prob": float(pred_prob),
                "top_pos_words": top_pos,
                "top_neg_words": top_neg,
                "probs": {labels[k]: float(probs[k]) for k in range(len(labels))},
                "margin": float(margin)
            })
    return results

# --- News pipeline ---
def fetch_robust_news_data(client, keywords, models):
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0'
    all_news, today = [], datetime.now()
    start_date = today - timedelta(days=14)
    for kw in keywords:
        st.write(f"▶ '{kw}' 키워드 뉴스 수집 중...")
        for lang, rss_url in {
            'ko': f"https://news.google.com/rss/search?q={quote(kw)}&hl=ko&gl=KR&ceid=KR:ko",
            'en': f"https://news.google.com/rss/search?q={quote(kw)}&hl=en-US&gl=US&ceid=US:en-US"
        }.items():
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:30]:
                title = entry.get('title', '')
                if not title: continue
                try: pub_date = pd.to_datetime(entry.get('published'))
                except: pub_date = None
                if pub_date and start_date <= pub_date <= today:
                    all_news.append({"Date": pub_date.date(), "Title": title, "Keyword": kw, "Language": lang})
    if not all_news:
        return pd.DataFrame()
    df = pd.DataFrame(all_news).drop_duplicates(subset=["Title","Keyword","Language"])
    rows = []
    for lang in df["Language"].unique():
        subset = df[df["Language"]==lang].reset_index(drop=True)
        titles = subset["Title"].tolist()
        model, tok = models[lang]["model"], models[lang]["tokenizer"]
        results = classify_with_conditional_ig(titles, model, tok)
        for i,r in enumerate(results):
            posp = r["probs"].get("positive", r["probs"].get("pos",0))
            negp = r["probs"].get("negative", r["probs"].get("neg",0))
            score = posp - negp
            rows.append({
                "날짜": subset.loc[i,"Date"],
                "Title": r["text"],
                "Label": r["label"],
                "Prob": r["prob"],
                "Top_Positive_Keywords": ", ".join(r["top_pos_words"]),
                "Top_Negative_Keywords": ", ".join(r["top_neg_words"]),
                "Keyword": subset.loc[i,"Keyword"],
                "Language": lang,
                "Sentiment": score,
                "InsertedAt": datetime.utcnow()
            })
    final_df = pd.DataFrame(rows)
    if client is not None and not final_df.empty:
        ensure_bq_table_schema(client, BQ_DATASET, BQ_TABLE_NEWS)
        pandas_gbq.to_gbq(final_df, f"{client.project}.{BQ_DATASET}.{BQ_TABLE_NEWS}",
                          project_id=client.project, if_exists="append",
                          credentials=client._credentials)
    return final_df

# --- Data Fetching & Processing Functions ---

@st.cache_data(ttl=3600)
def get_categories_from_bq(_client):
    """BigQuery에서 분석 가능한 품목 카테고리 목록을 가져옵니다."""
    project_id = _client.project
    table_id = f"{project_id}.data_explorer.tds_data"
    try:
        with st.spinner("BigQuery에서 카테고리 목록 불러오는 중..."):
            query = f"SELECT DISTINCT Category FROM `{table_id}` WHERE Category IS NOT NULL ORDER BY Category"
            df = _client.query(query).to_dataframe()
        return sorted(df['Category'].astype(str).unique())
    except Exception as e:
        st.error(f"BigQuery 테이블({table_id})을 읽는 중 오류: {e}")
        return []

def get_trade_data_from_bq(client, categories):
    """선택된 카테고리에 대한 수출입 데이터를 BigQuery에서 가져옵니다."""
    if not categories: return pd.DataFrame()
    project_id = client.project
    table_id = f"{project_id}.data_explorer.tds_data"
    try:
        query_params = [bigquery.ArrayQueryParameter("categories", "STRING", categories)]
        sql = f"SELECT * FROM `{table_id}` WHERE Category IN UNNEST(@categories)"
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        with st.spinner(f"BigQuery에서 {len(categories)}개 카테고리 데이터 로드 중..."):
            df = client.query(sql, job_config=job_config).to_dataframe()
        
        # 데이터 타입 정리
        for col in df.columns:
            if 'price' in col.lower() or 'value' in col.lower() or 'volume' in col.lower():
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"BigQuery에서 TDS 데이터 읽는 중 오류: {e}")
        return pd.DataFrame()

def deduplicate_and_write_to_bq(client, df_new, table_name, subset_cols=None):
    """데이터를 BigQuery에 중복 제거하여 저장합니다."""
    project_id = client.project
    table_id = f"{project_id}.data_explorer.{table_name}"
    try:
        try:
            sql = f"SELECT * FROM `{table_id}`"
            df_existing = client.query(sql).to_dataframe()
        except Exception: 
            df_existing = pd.DataFrame()

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        if subset_cols:
            df_deduplicated = df_combined.drop_duplicates(subset=subset_cols, keep='last')
        else:
            df_deduplicated = df_combined.drop_duplicates(keep='last')
        
        with st.spinner(f"'{table_name}' 테이블에 데이터 저장 중..."):
            pandas_gbq.to_gbq(df_deduplicated, table_id, project_id=project_id, if_exists="replace", credentials=client._credentials)
        st.sidebar.success(f"'{table_name}'에 데이터 업데이트 완료.")
    except Exception as e: 
        st.error(f"BigQuery 저장 중 오류: {e}")

def call_naver_api(url, body, naver_keys):
    """네이버 API를 호출하고 결과를 반환합니다."""
    try:
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", naver_keys['id'])
        request.add_header("X-Naver-Client-Secret", naver_keys['secret'])
        request.add_header("Content-Type", "application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        if response.getcode() == 200:
            return json.loads(response.read().decode('utf-8'))
        return None
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        st.error(f"Naver API 오류 발생: {e.code} - {e.reason}")
        st.error(f"서버 응답: {error_body}")
        return None
    except Exception as e:
        st.error(f"API 호출 중 알 수 없는 오류 발생: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_naver_trends_data(keywords, start_date, end_date, naver_keys):
    """
    네이버 데이터랩(검색어 트렌드, 쇼핑인사이트) 데이터를 가져옵니다.
    키워드에 따라 자동으로 쇼핑인사이트 카테고리 ID를 매칭합니다.
    """
    if (end_date - start_date).days > 90:
        st.sidebar.error("네이버 트렌드 조회 기간은 최대 3개월(90일)입니다.")
        return pd.DataFrame()

    NAVER_SHOPPING_CAT_MAP = {
        '아보카도': "50000007", '바나나': "50000007", '사과': "50000007", '수입과일': "50000007",
        '커피': "50000004", '커피 생두': "50000004",
        '쌀': "50000006", '고등어': "50000009"
    }
    all_data = []

    for keyword in keywords:
        keyword_dfs = []
        with st.spinner(f"'{keyword}' 네이버 트렌드 데이터를 가져오는 중..."):
            if naver_keys['id'] and naver_keys['secret']:
                # 1. 네이버 검색어 트렌드 API 호출 (이 부분은 구조가 원래 맞았음)
                body_search = json.dumps({
                    "startDate": start_date.strftime('%Y-%m-%d'), 
                    "endDate": end_date.strftime('%Y-%m-%d'), 
                    "timeUnit": "date", 
                    "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
                })
                search_res = call_naver_api("https://openapi.naver.com/v1/datalab/search", body_search, naver_keys)
                if search_res and search_res.get('results'):
                    df_search = pd.DataFrame(search_res['results'][0]['data'])
                    keyword_dfs.append(df_search.rename(columns={'period': '날짜', 'ratio': f'NaverSearch_{keyword}'}))

                # 2. 네이버 쇼핑인사이트 API 호출
                lower_keyword = keyword.lower().replace(' ', '')
                if lower_keyword in NAVER_SHOPPING_CAT_MAP:
                    category_id = NAVER_SHOPPING_CAT_MAP[lower_keyword]
                    
                    # [수정] API 문서에 맞는 배열/객체 형태로 구조 변경
                    body_shop = json.dumps({
                        "startDate": start_date.strftime('%Y-%m-%d'),
                        "endDate": end_date.strftime('%Y-%m-%d'),
                        "timeUnit": "date",
                        "category": [{"name": keyword, "param": [category_id]}],
                        "keyword": [{"name": keyword, "param": [keyword]}]
                    })
                    
                    shop_res = call_naver_api("https://openapi.naver.com/v1/datalab/shopping/categories", body_shop, naver_keys)
                    if shop_res and shop_res.get('results'):
                        df_shop = pd.DataFrame(shop_res['results'][0]['data'])
                        keyword_dfs.append(df_shop.rename(columns={'period': '날짜', 'ratio': f'NaverShop_{keyword}'}))
                else:
                    st.sidebar.info(f"'{keyword}'에 대한 쇼핑인사이트 카테고리 정보가 없어 검색 트렌드만 조회합니다.")

            if keyword_dfs:
                for df in keyword_dfs:
                    df['날짜'] = pd.to_datetime(df['날짜'])
                merged_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), keyword_dfs)
                all_data.append(merged_df)

    if not all_data: 
        return pd.DataFrame()
        
    return reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_data)

def fetch_kamis_data(client, item_info, start_date, end_date, kamis_keys):
    """KAMIS에서 기간별 도매 가격 데이터를 한 번의 API 호출로 가져옵니다."""
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    item_code, kind_code = item_info['item_code'], item_info['kind_code']

    with st.spinner("KAMIS에서 기간 데이터 조회 중..."):
        url = (
            "http://www.kamis.or.kr/service/price/xml.do?action=periodWholesaleProductList"
            f"&p_product_cls_code=01&p_startday={start_str}&p_endday={end_str}"
            f"&p_item_category_code={item_info['cat_code']}&p_item_code={item_code}&p_kind_code={kind_code}"
            f"&p_product_rank_code={item_info['rank_code']}&p_convert_kg_yn=Y"
            f"&p_cert_key={kamis_keys['key']}&p_cert_id={kamis_keys['id']}&p_returntype=json"
        )
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data and "data" in data and data["data"] and "item" in data["data"]:
                    price_data = data["data"]["item"]
                    if not price_data:
                        st.sidebar.warning("해당 기간에 대한 KAMIS 데이터가 없습니다.")
                        return pd.DataFrame()
                    
                    df_new = pd.DataFrame(price_data).rename(columns={'regday': '날짜', 'price': '도매가격_원'})
                    df_new = df_new[['날짜', '도매가격_원']]
                    
                    def format_date(date_str):
                        return f"{start_date.year}-{date_str}" if len(date_str) <= 5 else date_str
                    
                    df_new['날짜'] = pd.to_datetime(df_new['날짜'].apply(format_date))
                    df_new['도매가격_원'] = pd.to_numeric(df_new['도매가격_원'].str.replace(',', ''), errors='coerce')
                    return df_new
                else:
                    st.sidebar.warning("해당 기간에 대한 KAMIS 데이터가 없습니다.")
            else:
                st.sidebar.error(f"KAMIS 서버 응답 오류: Status {response.status_code}")

        except Exception as e:
            st.sidebar.error(f"KAMIS API 호출 중 오류: {e}")

    return pd.DataFrame()

def fetch_robust_news_data(client, keywords, models):
    """안정성을 높인 뉴스 데이터 수집 및 분석 함수 (RSS 기반)"""
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    
    all_news_data, today = [], datetime.now()
    start_date = today - timedelta(days=14)

    for keyword in keywords:
        st.write(f"▶ '{keyword}' 키워드 뉴스 수집 중...")
        urls = {
            'ko': f"https://news.google.com/rss/search?q={quote(keyword)}&hl=ko&gl=KR&ceid=KR:ko",
            'en': f"https://news.google.com/rss/search?q={quote(keyword)}&hl=en-US&gl=US&ceid=US:en-US"
        }
        for lang, rss_url in urls.items():
            try:
                feed = feedparser.parse(rss_url)
                if not feed.entries:
                    st.warning(f"'{keyword}'({lang}) RSS 피드에서 기사를 찾지 못했습니다.")
                    continue
                
                model, tokenizer = models[lang]['model'], models[lang]['tokenizer']
                for entry in feed.entries[:20]:
                    try:
                        article = Article(entry.link, config=config, language=lang)
                        article.download(); article.parse()

                        if not article.publish_date: continue
                        pub_date = article.publish_date.replace(tzinfo=None)

                        if start_date <= pub_date <= today:
                            title = article.title
                            pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                            analysis = pipe(title)[0]
                            label, score = analysis['label'], analysis['score']
                            sentiment_score = score if label.lower() in ['positive', '5 stars'] else -score
                            
                            all_news_data.append({'Date': pub_date.date(), 'Title': title, 'Sentiment': sentiment_score, 'Keyword': keyword, 'Language': lang})
                    except Exception as e:
                        st.warning(f"기사 처리 중 오류: {entry.link} - 원인: {str(e)[:100]}")
            except Exception as e:
                st.error(f"RSS 피드 처리 중 오류: {rss_url} - 원인: {e}")

    if not all_news_data:
        st.sidebar.error("수집된 뉴스가 없습니다. 잠시 후 다시 시도해보세요.")
        return pd.DataFrame()

    final_df = pd.DataFrame(all_news_data)
    deduplicate_and_write_to_bq(client, final_df, "news_sentiment_cache", subset_cols=['Title', 'Keyword', 'Language'])
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    return final_df.rename(columns={'Date': '날짜'})

# --- Constants & App ---
KAMIS_FULL_DATA = {
    '쌀': {'cat_code': '100', 'item_code': '111', 'kinds': {'20kg': '01', '백미': '02'}},
    '감자': {'cat_code': '100', 'item_code': '152', 'kinds': {'수미(노지)': '01', '수미(시설)': '04'}},
    '배추': {'cat_code': '200', 'item_code': '211', 'kinds': {'봄': '01', '여름': '02', '가을': '03'}},
    '양파': {'cat_code': '200', 'item_code': '245', 'kinds': {'양파': '00', '햇양파': '02'}},
    '사과': {'cat_code': '400', 'item_code': '411', 'kinds': {'후지': '05', '아오리': '06'}},
    '바나나': {'cat_code': '400', 'item_code': '418', 'kinds': {'수입': '02'}},
    '아보카도': {'cat_code': '400', 'item_code': '430', 'kinds': {'수입': '00'}},
    '고등어': {'cat_code': '600', 'item_code': '611', 'kinds': {'생선': '01', '냉동': '02'}},
}
st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 분석 대시보드")

bq_client = get_bq_connection()
sentiment_assets = load_sentiment_assets()
if bq_client is None: st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ 분석 설정")
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'categories' not in st.session_state: st.session_state.categories = get_categories_from_bq(bq_client)

st.sidebar.subheader("1. 분석 대상 설정")
if not st.session_state.categories:
    st.sidebar.warning("BigQuery에 데이터가 없습니다. 새 데이터를 추가해주세요.")
else:
    selected_categories = st.sidebar.multiselect("분석할 품목 카테고리 선택", st.session_state.categories, default=st.session_state.get('selected_categories', []))
    if st.sidebar.button("🚀 선택 완료 및 분석 시작"):
        if not selected_categories:
            st.sidebar.warning("카테고리를 선택해주세요.")
        else:
            st.session_state.raw_trade_df = get_trade_data_from_bq(bq_client, selected_categories)
            if st.session_state.raw_trade_df is not None and not st.session_state.raw_trade_df.empty:
                st.session_state.data_loaded = True
                st.session_state.selected_categories = selected_categories
            else:
                st.sidebar.error("데이터를 불러오지 못했습니다.")

if not st.session_state.data_loaded:
    st.info("👈 사이드바에서 분석할 카테고리를 선택하고 '분석 시작' 버튼을 눌러주세요.")
    st.stop()

# --- Main App Logic ---
raw_trade_df = st.session_state.raw_trade_df
selected_categories = st.session_state.selected_categories
st.sidebar.success(f"**{', '.join(selected_categories)}** 데이터 로드 완료!")
st.sidebar.markdown("---")

try:
    raw_trade_df.dropna(subset=['Date'], inplace=True)
    file_start_date, file_end_date = raw_trade_df['Date'].min(), raw_trade_df['Date'].max()
    
    st.sidebar.subheader("2. 분석 기간 및 키워드 설정")
    start_date = pd.to_datetime(st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date))
    end_date = pd.to_datetime(st.sidebar.date_input('종료일', file_end_date, min_value=start_date, max_value=file_end_date))
    
    default_keywords = ", ".join(selected_categories)
    keyword_input = st.sidebar.text_input("뉴스/트렌드 분석 키워드", default_keywords)
    search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
except Exception as e:
    st.error(f"데이터 기간 설정 중 오류: {e}")
    st.stop()

st.sidebar.subheader("3. 외부 데이터 연동")

st.sidebar.markdown("##### KAMIS 농산물 가격")
kamis_api_key = st.sidebar.text_input("KAMIS API Key", type="password", help="KAMIS 공공데이터 포털에서 발급받은 인증키")
kamis_api_id = st.sidebar.text_input("KAMIS API ID", type="password", help="KAMIS API 신청 시 등록한 ID")
item_name = st.sidebar.selectbox("품목 선택", list(KAMIS_FULL_DATA.keys()))
if item_name:
    kind_name = st.sidebar.selectbox("품종 선택", list(KAMIS_FULL_DATA[item_name]['kinds'].keys()))
    if st.sidebar.button("KAMIS 데이터 가져오기"):
        if kamis_api_key and kamis_api_id:
            item_info = {**KAMIS_FULL_DATA[item_name], 'item_name': item_name, 'kind_name': kind_name, 'rank_code': '01'}
            item_info['kind_code'] = KAMIS_FULL_DATA[item_name]['kinds'][kind_name]
            st.session_state.wholesale_data = fetch_kamis_data(bq_client, item_info, start_date, end_date, {'key': kamis_api_key, 'id': kamis_api_id})
        else:
            st.sidebar.error("KAMIS API Key와 ID를 모두 입력해주세요.")

st.sidebar.markdown("##### 네이버 트렌드 데이터")
naver_client_id = st.sidebar.text_input("Naver API Client ID", type="password")
naver_client_secret = st.sidebar.text_input("Naver API Client Secret", type="password")
if st.sidebar.button("네이버 트렌드 가져오기"):
    if not search_keywords: st.sidebar.warning("분석할 키워드를 먼저 입력해주세요.")
    else:
        st.session_state.search_data = fetch_naver_trends_data(search_keywords, start_date, end_date, {'id': naver_client_id, 'secret': naver_client_secret})

st.sidebar.markdown("##### 뉴스 감성 분석")
if st.sidebar.button("최신 뉴스 분석하기"):
    if not search_keywords: st.sidebar.warning("분석할 키워드를 먼저 입력해주세요.")
    else:
        st.session_state.news_data = fetch_robust_news_data(bq_client, search_keywords, sentiment_assets)

# --- Main Display Tabs ---
raw_wholesale_df = st.session_state.get('wholesale_data', pd.DataFrame())
raw_search_df = st.session_state.get('search_data', pd.DataFrame())
raw_news_df = st.session_state.get('news_data', pd.DataFrame())

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1️⃣ 원본 데이터", "2️⃣ 데이터 표준화", "3️⃣ 뉴스 감성 분석", "4️⃣ 상관관계 분석", "📈 시계열 예측"])

with tab1:
    st.subheader("A. 수출입 데이터 (선택 기간)"); st.dataframe(raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)])
    st.subheader("B. 외부 가격 데이터"); st.dataframe(raw_wholesale_df)
    st.subheader("C. 트렌드 데이터"); st.dataframe(raw_search_df)
    st.subheader("D. 뉴스 데이터"); st.dataframe(raw_news_df)
    
with tab2:
    st.header("데이터 표준화: 주별(Weekly) 데이터로 변환")
    trade_df_in_range = raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)]
    filtered_trade_df = trade_df_in_range[trade_df_in_range['Category'].isin(selected_categories)].copy()
    
    if not filtered_trade_df.empty:
        filtered_trade_df.set_index('Date', inplace=True)
        trade_weekly = filtered_trade_df.resample('W-Mon').agg(수입액_USD=('Value', 'sum'), 수입량_KG=('Volume', 'sum')).copy()
        trade_weekly['수입단가_USD_KG'] = trade_weekly['수입액_USD'] / trade_weekly['수입량_KG']
        trade_weekly.index.name = '날짜'
        
        dfs_to_process = {
            'wholesale': raw_wholesale_df,
            'search': raw_search_df,
            'news': raw_news_df
        }
        weekly_dfs = {'trade': trade_weekly}

        for name, df in dfs_to_process.items():
            if not df.empty and '날짜' in df.columns:
                df['날짜'] = pd.to_datetime(df['날짜'])
                df_in_range = df[(df['날짜'] >= start_date) & (df['날짜'] <= end_date)]
                if not df_in_range.empty:
                    df_weekly = df_in_range.set_index('날짜').resample('W-Mon').mean(numeric_only=True)
                    df_weekly.index.name = '날짜'
                    weekly_dfs[name] = df_weekly

        st.session_state['weekly_dfs'] = weekly_dfs
        st.write("### 주별 집계 데이터 샘플")
        for name, df in weekly_dfs.items():
            st.write(f"##### {name.capitalize()} Data (Weekly)")
            st.dataframe(df.head())
    else:
        st.warning("선택된 기간에 해당하는 수출입 데이터가 없습니다.")

with tab3:
    st.header("뉴스 감성 분석 결과")
    if not raw_news_df.empty:
        news_weekly = st.session_state.get('weekly_dfs', {}).get('news', pd.DataFrame())
        if not news_weekly.empty:
            fig = px.line(news_weekly, y='Sentiment', title="주별 평균 뉴스 감성 점수", labels={'Sentiment': '감성 점수'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("수집된 뉴스 기사 목록 (최신순)")
        st.dataframe(raw_news_df.sort_values(by='날짜', ascending=False))
    else:
        st.info("사이드바에서 '최신 뉴스 분석하기' 버튼을 눌러주세요.")

with tab4:
    st.header("상관관계 분석")
    weekly_dfs = st.session_state.get('weekly_dfs', {})
    if weekly_dfs:
        dfs_to_concat = [df for df in weekly_dfs.values() if not df.empty]
        if dfs_to_concat:
            final_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), dfs_to_concat)
            final_df = final_df.interpolate(method='linear', limit_direction='both').dropna(how='all')
            st.session_state['final_df'] = final_df
            
            st.subheader("통합 데이터 시각화")
            df_long = final_df.reset_index().melt(id_vars='날짜', var_name='데이터 종류', value_name='값')
            fig = px.line(df_long, x='날짜', y='값', color='데이터 종류', title="통합 데이터 시계열 추이")
            st.plotly_chart(fig, use_container_width=True)

            if len(final_df.columns) > 1:
                st.markdown("---")
                st.subheader("상관관계 히트맵")
                corr_matrix = final_df.corr(numeric_only=True)
                fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1])
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("분석할 주별 데이터가 없습니다.")
    else:
        st.warning("2단계에서 데이터가 처리되지 않았습니다.")

with tab5:
    st.header("시계열 분해 및 예측 (by Prophet)")
    final_df = st.session_state.get('final_df', pd.DataFrame())
    if not final_df.empty:
        forecast_col = st.selectbox("예측 대상 변수 선택", final_df.columns)
        if forecast_col:
            ts_data = final_df[[forecast_col]].dropna()
            if len(ts_data) < 24:
                st.warning(f"시계열 분석을 위해 최소 24주 이상의 데이터가 필요합니다. 현재 데이터: {len(ts_data)}주")
            else:
                st.subheader(f"'{forecast_col}' 시계열 분해")
                period = 52 if len(ts_data) >= 104 else int(len(ts_data) / 2)
                decomposition = seasonal_decompose(ts_data[forecast_col], model='additive', period=period)
                
                fig_decompose = go.Figure()
                fig_decompose.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'))
                fig_decompose.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                fig_decompose.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                st.plotly_chart(fig_decompose, use_container_width=True)

                st.subheader(f"'{forecast_col}' 미래 12주 예측")
                prophet_df = ts_data.reset_index().rename(columns={'날짜': 'ds', forecast_col: 'y'})
                
                m = Prophet()
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=12, freq='W')
                forecast = m.predict(future)

                fig_forecast = plot_plotly(m, forecast)
                fig_forecast.update_layout(title=f"'{forecast_col}' 미래 예측", xaxis_title='날짜', yaxis_title='예측값')
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.write("#### 예측 데이터 테이블")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))
    else:
        st.info("4번 탭에서 데이터가 통합되어야 예측을 수행할 수 있습니다.")
