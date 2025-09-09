# total_ml_pred.py  —  2025-09-09 업데이트
# 요청 반영: 다중 모델 교차 분석, AI 트렌드 요약 카드, Naver 중심 연동, newspaper3k 안정화, 메인화면 초기화/재로딩 버튼

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, timezone
from functools import reduce
import json
import urllib.request
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas_gbq
import feedparser
from urllib.parse import quote
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.plot import plot_plotly
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# (NEW) optional: robust article text fetch
try:
    from newspaper import Article, Config as NewsConfig
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False

# --- 1. Constants and Configuration ---
BQ_DATASET = "data_explorer"
BQ_TABLE_NAVER = "naver_trends_cache"
BQ_TABLE_NEWS = "news_sentiment_finbert"  # 기존 테이블명 유지 (스키마 확장)

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

# --- 2. GCP Connection and Helper Functions ---
@st.cache_resource
def get_bq_connection():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        client = bigquery.Client(credentials=creds, project=creds.project_id)
        return client
    except Exception as e:
        st.error(f"Google BigQuery 연결 실패: secrets.toml을 확인하세요. 오류: {e}")
        return None

def _safe_update_news_table_schema(client):
    """기존 테이블에 다중 모델 컬럼을 추가(없을 경우)"""
    try:
        project_id = client.project
        table_id = f"{project_id}.{BQ_DATASET}.{BQ_TABLE_NEWS}"
        table = client.get_table(table_id)
        existing = {s.name for s in table.schema}
        add_fields = []
        wanted = [
            ("FinBERT_Sentiment", "FLOAT"),
            ("FinBERT_Label", "STRING"),
            ("KLUE_Sentiment", "FLOAT"),
            ("KLUE_Label", "STRING"),
            ("NSMC_Sentiment", "FLOAT"),
            ("NSMC_Label", "STRING"),
            ("RawUrl", "STRING"),
        ]
        for name, t in wanted:
            if name not in existing:
                add_fields.append(bigquery.SchemaField(name, t))
        if add_fields:
            new_schema = list(table.schema) + add_fields
            table.schema = new_schema
            client.update_table(table, ["schema"])
    except Exception:
        pass  # 스키마 확장은 실패해도 앱 동작에는 치명적이지 않게

def ensure_news_table_exists(client):
    project_id = client.project
    full_table_id = f"{project_id}.{BQ_DATASET}.{BQ_TABLE_NEWS}"
    try:
        client.get_table(full_table_id)
        _safe_update_news_table_schema(client)
    except Exception:
        st.write(f"뉴스 분석 테이블 '{full_table_id}'을 새로 생성합니다.")
        schema = [
            bigquery.SchemaField("날짜", "DATE"),
            bigquery.SchemaField("Title", "STRING"),
            bigquery.SchemaField("Keyword", "STRING"),
            # (KEEP for backward)
            bigquery.SchemaField("Sentiment", "FLOAT"),
            bigquery.SchemaField("Label", "STRING"),
            bigquery.SchemaField("InsertedAt", "TIMESTAMP"),
            # (NEW) multi-model fields
            bigquery.SchemaField("FinBERT_Sentiment", "FLOAT"),
            bigquery.SchemaField("FinBERT_Label", "STRING"),
            bigquery.SchemaField("KLUE_Sentiment", "FLOAT"),
            bigquery.SchemaField("KLUE_Label", "STRING"),
            bigquery.SchemaField("NSMC_Sentiment", "FLOAT"),
            bigquery.SchemaField("NSMC_Label", "STRING"),
            bigquery.SchemaField("RawUrl", "STRING"),
        ]
        table = bigquery.Table(full_table_id, schema=schema)
        client.create_table(table)
        st.success(f"테이블 '{BQ_TABLE_NEWS}' 생성 완료.")

def call_naver_api(url, body, naver_keys):
    try:
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", naver_keys['id'])
        request.add_header("X-Naver-Client-Secret", naver_keys['secret'])
        request.add_header("Content-Type", "application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        if response.getcode() == 200:
            return json.loads(response.read().decode('utf-8'))
        return None
    except Exception as e:
        st.error(f"Naver API 오류 발생: {e}")
        return None

# --- 3. Sentiment Models (Multi-model) ---
@st.cache_resource
def load_models():
    models = {}
    # 1) 금융 특화
    try:
        with st.spinner("모델 로드: KR-FinBERT ..."):
            name = "snunlp/KR-FinBERT-SC"
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            models["finbert"] = pipeline("sentiment-analysis", model=mdl, tokenizer=tok)
    except Exception as e:
        st.warning(f"KR-FinBERT 로드 실패: {e}")
        models["finbert"] = None

    # 2) 범용(klue/bert-base) — 미세조정 모델이 아니어 실패 가능 → 안전장치
    try:
        with st.spinner("모델 로드: KLUE/BERT (범용) ..."):
            name = "klue/bert-base"
            tok = AutoTokenizer.from_pretrained(name)
            # text-classification 헤드가 없으면 AutoModelForSequenceClassification에서 실패할 수 있음
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            models["klue"] = pipeline("sentiment-analysis", model=mdl, tokenizer=tok)
    except Exception as e:
        st.warning(f"KLUE/BERT 로드 실패(범용): {e}")
        models["klue"] = None

    # 3) 상품평(NSMC)
    try:
        with st.spinner("모델 로드: NSMC(상품평) ..."):
            name = "heew/bert-nsmc-base-v1"
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            models["nsmc"] = pipeline("sentiment-analysis", model=mdl, tokenizer=tok)
    except Exception as e:
        st.warning(f"NSMC 로드 실패: {e}")
        models["nsmc"] = None

    return models

def _label_score_to_signed(pred):
    """Transformers 결과를 통일된 (-1~+1)로 정규화"""
    if pred is None:
        return 0.0, "neutral"
    lbl = str(pred.get("label", "")).lower()
    score = float(pred.get("score", 0.0))
    if "neg" in lbl:  # negative
        return -score, "negative"
    if "pos" in lbl:  # positive
        return +score, "positive"
    if "neu" in lbl:
        return 0.0, "neutral"
    # 알 수 없는 레이블: 0 처리
    return 0.0, lbl or "neutral"

def analyze_sentiment_multi(texts, models):
    """각 문장에 대해 세 모델 점수 동시 계산"""
    results = []
    if not texts:
        return results
    # batch inference per model (성능)
    preds = {}
    for key in ["finbert", "klue", "nsmc"]:
        clf = models.get(key)
        if clf is None:
            preds[key] = [None] * len(texts)
            continue
        try:
            preds[key] = clf(texts)
        except Exception:
            preds[key] = [None] * len(texts)

    for i, t in enumerate(texts):
        finbert_s, finbert_l = _label_score_to_signed(preds["finbert"][i] if preds["finbert"] else None)
        klue_s, klue_l = _label_score_to_signed(preds["klue"][i] if preds["klue"] else None)
        nsmc_s, nsmc_l = _label_score_to_signed(preds["nsmc"][i] if preds["nsmc"] else None)
        # 기존 단일 Sentiment/Label 필드 호환을 위해 FinBERT 중심 유지
        results.append({
            "FinBERT_Sentiment": finbert_s, "FinBERT_Label": finbert_l,
            "KLUE_Sentiment": klue_s, "KLUE_Label": klue_l,
            "NSMC_Sentiment": nsmc_s, "NSMC_Label": nsmc_l,
            "Sentiment": finbert_s, "Label": finbert_l
        })
    return results

def _fetch_article_text(url):
    """newspaper3k로 본문 확보 시도(안정화 옵션)"""
    if not _HAS_NEWSPAPER or not url:
        return ""
    try:
        cfg = NewsConfig()
        cfg.browser_user_agent = "Mozilla/5.0 (compatible; NewsBot/1.0; +https://example.com/bot)"
        cfg.request_timeout = 10
        art = Article(url, language='ko', config=cfg)
        art.download()
        art.parse()
        return art.text or ""
    except Exception:
        return ""

def get_news_with_multi_model_analysis(_bq_client, models, keyword, days_limit=7):
    """RSS 기반 수집 + (가능시) 본문 fetch → 세 모델 동시 감성 분석 + BQ 캐시 저장"""
    project_id = _bq_client.project
    full_table_id = f"{project_id}.{BQ_DATASET}.{BQ_TABLE_NEWS}"

    # 1) 캐시 조회
    try:
        time_limit = datetime.now(timezone.utc) - timedelta(days=days_limit)
        query = f"""
        SELECT * FROM `{full_table_id}`
        WHERE Keyword = @keyword AND InsertedAt >= @time_limit
        ORDER BY 날짜 DESC
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("keyword", "STRING", keyword),
            bigquery.ScalarQueryParameter("time_limit", "TIMESTAMP", time_limit)
        ])
        df_cache = _bq_client.query(query, job_config=job_config).to_dataframe()
    except Exception:
        df_cache = pd.DataFrame()

    if not df_cache.empty:
        st.sidebar.success(f"✔️ '{keyword}' 최신 분석 결과를 캐시에서 로드했습니다.")
        return df_cache

    st.sidebar.warning(f"'{keyword}' 최신 캐시 없음 → 새로 수집·분석합니다...")

    # 2) RSS 수집 (Google News, 국내 설정)
    all_news = []
    rss_url = f"https://news.google.com/rss/search?q={quote(keyword)}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)

    for entry in feed.entries[:30]:
        title = entry.get('title', '').strip()
        link = entry.get('link', '').strip()
        if not title:
            continue
        try:
            pub_date = pd.to_datetime(entry.get('published')).date()
        except Exception:
            pub_date = datetime.utcnow().date()

        # (NEW) 기사 본문 일부 확보(안정성 옵션)
        body = _fetch_article_text(link)
        # 제목과 본문을 합쳐 모델에 입력(본문 없으면 제목만)
        text_for_model = (title + " " + body).strip() if body else title

        all_news.append({"날짜": pub_date, "Title": title, "RawUrl": link, "ModelInput": text_for_model})

    if not all_news:
        st.error(f"'{keyword}'에 대한 뉴스를 찾지 못했습니다.")
        return pd.DataFrame()

    df_new = pd.DataFrame(all_news).drop_duplicates(subset=["Title"])
    with st.spinner(f"다중 모델로 '{keyword}' 뉴스 감성 분석 중..."):
        multi = analyze_sentiment_multi(df_new['ModelInput'].tolist(), models)

    multi_df = pd.DataFrame(multi)
    df_new = pd.concat([df_new.drop(columns=['ModelInput']), multi_df], axis=1)
    df_new['Keyword'] = keyword
    df_new['InsertedAt'] = datetime.now(timezone.utc)

    # (캐시 저장 - 스키마 확장 반영)
    try:
        df_to_gbq = df_new[[
            "날짜", "Title", "Keyword", "Sentiment", "Label", "InsertedAt",
            "FinBERT_Sentiment", "FinBERT_Label",
            "KLUE_Sentiment", "KLUE_Label",
            "NSMC_Sentiment", "NSMC_Label",
            "RawUrl"
        ]]
        pandas_gbq.to_gbq(df_to_gbq, full_table_id, project_id=project_id,
                          if_exists="append", credentials=_bq_client._credentials)
    except Exception as e:
        st.sidebar.warning(f"BigQuery 저장 실패(계속 진행): {e}")
        df_to_gbq = df_new  # 화면 표시는 계속

    return df_to_gbq

# --- 4. Data Fetching (BigQuery, KAMIS, NAVER) ---
@st.cache_data(ttl=3600)
def get_categories_from_bq(_client):
    project_id = _client.project
    table_id = f"{project_id}.{BQ_DATASET}.tds_data"
    try:
        query = f"SELECT DISTINCT Category FROM `{table_id}` WHERE Category IS NOT NULL ORDER BY Category"
        df = _client.query(query).to_dataframe()
        return sorted(df['Category'].astype(str).unique())
    except Exception:
        return []

def get_trade_data_from_bq(client, categories):
    if not categories:
        return pd.DataFrame()
    project_id = client.project
    table_id = f"{project_id}.{BQ_DATASET}.tds_data"
    try:
        query_params = [bigquery.ArrayQueryParameter("categories", "STRING", categories)]
        sql = f"SELECT * FROM `{table_id}` WHERE Category IN UNNEST(@categories)"
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        df = client.query(sql, job_config=job_config).to_dataframe()
        for col in df.columns:
            if 'price' in col.lower() or 'value' in col.lower() or 'volume' in col.lower():
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"BigQuery에서 TDS 데이터 읽는 중 오류: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_naver_trends_data(_client, keywords, start_date, end_date, naver_keys):
    # Google Trends 제거, Naver DataLab 집중 (검색어 + 쇼핑)
    project_id = _client.project
    table_id = f"{project_id}.{BQ_DATASET}.{BQ_TABLE_NAVER}"
    try:
        sql = f"SELECT * FROM `{table_id}` ORDER BY 날짜"
        df_cache = _client.query(sql).to_dataframe()
        if not df_cache.empty:
            df_cache['날짜'] = pd.to_datetime(df_cache['날짜'])
    except Exception:
        df_cache = pd.DataFrame(columns=['날짜'])

    fetch_start_date = start_date
    if not df_cache.empty:
        last_cached_date = df_cache['날짜'].max()
        if start_date > last_cached_date:
            fetch_start_date = start_date
        elif end_date > last_cached_date:
            fetch_start_date = last_cached_date + timedelta(days=1)
        else:
            fetch_start_date = end_date + timedelta(days=1)

    new_data_list = []
    if fetch_start_date <= end_date:
        current_start = fetch_start_date
        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=89), end_date)
            NAVER_SHOPPING_CAT_MAP = {
                '아보카도': "50000007", '바나나': "50000007", '사과': "50000007",
                '커피': "50000004", '쌀': "50000006", '고등어': "50000009"
            }
            all_data_chunk = []
            for keyword in keywords:
                keyword_dfs = []

                # 검색어 인덱스
                body_search = json.dumps({
                    "startDate": current_start.strftime('%Y-%m-%d'),
                    "endDate": current_end.strftime('%Y-%m-%d'),
                    "timeUnit": "date",
                    "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
                })
                search_res = call_naver_api("https://openapi.naver.com/v1/datalab/search", body_search, naver_keys)
                if search_res and search_res.get('results') and search_res['results'][0]['data']:
                    df_search = pd.DataFrame(search_res['results'][0]['data'])
                    if not df_search.empty:
                        keyword_dfs.append(df_search.rename(columns={'period': '날짜', 'ratio': f'NaverSearch_{keyword}'}))

                # 쇼핑 인사이트
                norm_keyword = keyword.lower().replace(' ', '')
                if norm_keyword in NAVER_SHOPPING_CAT_MAP:
                    cat_id = NAVER_SHOPPING_CAT_MAP[norm_keyword]
                    body_shop = json.dumps({
                        "startDate": current_start.strftime('%Y-%m-%d'),
                        "endDate": current_end.strftime('%Y-%m-%d'),
                        "timeUnit": "date",
                        "category": [{"name": keyword, "param": [cat_id]}],
                        "keyword": [{"name": keyword, "param": [keyword]}]
                    })
                    shop_res = call_naver_api("https://openapi.naver.com/v1/datalab/shopping/categories", body_shop, naver_keys)
                    if shop_res and shop_res.get('results') and shop_res['results'][0]['data']:
                        df_shop = pd.DataFrame(shop_res['results'][0]['data'])
                        if not df_shop.empty:
                            keyword_dfs.append(df_shop.rename(columns={'period': '날짜', 'ratio': f'NaverShop_{keyword}'}))

                if keyword_dfs:
                    for df in keyword_dfs:
                        df['날짜'] = pd.to_datetime(df['날짜'])
                    merged = reduce(lambda l, r: pd.merge(l, r, on='날짜', how='outer'), keyword_dfs)
                    all_data_chunk.append(merged)

            if all_data_chunk:
                new_data_list.append(reduce(lambda l, r: pd.merge(l, r, on='날짜', how='outer'), all_data_chunk))

            current_start = current_end + timedelta(days=1)
    else:
        st.sidebar.success("✔️ 네이버 트렌드: 모든 데이터가 캐시에 있습니다.")

    if new_data_list:
        non_empty = [df for df in new_data_list if not df.empty]
        if non_empty:
            df_new = pd.concat(non_empty, ignore_index=True)
            df_new['날짜'] = pd.to_datetime(df_new['날짜'])
            df_combined = pd.concat([df_cache, df_new], ignore_index=True)\
                .drop_duplicates(subset=['날짜'], keep='last')\
                .sort_values(by='날짜').reset_index(drop=True)
            pandas_gbq.to_gbq(df_combined, table_id, project_id=project_id,
                              if_exists="replace", credentials=_client._credentials)
            df_final = df_combined
        else:
            df_final = df_cache
    else:
        df_final = df_cache

    if df_final.empty:
        return pd.DataFrame()
    return df_final[(df_final['날짜'] >= start_date) & (df_final['날짜'] <= end_date)].reset_index(drop=True)

def fetch_kamis_data(_client, item_info, start_date, end_date, kamis_keys):
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    url = (f"http://www.kamis.or.kr/service/price/xml.do?action=periodWholesaleProductList"
           f"&p_product_cls_code=01&p_startday={start_str}&p_endday={end_str}"
           f"&p_item_category_code={item_info['cat_code']}&p_item_code={item_info['item_code']}&p_kind_code={item_info['kind_code']}"
           f"&p_product_rank_code={item_info['rank_code']}&p_convert_kg_yn=Y"
           f"&p_cert_key={kamis_keys['key']}&p_cert_id={kamis_keys['id']}&p_returntype=json")
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 KAMISClient/1.0"})
        if response.status_code == 200 and "data" in response.json() and "item" in response.json()["data"]:
            price_data = response.json()["data"]["item"]
            if not price_data:
                return pd.DataFrame()
            df_new = pd.DataFrame(price_data)[['regday', 'price']].rename(columns={'regday': '날짜', 'price': '도매가격_원'})
            def format_kamis_date(date_str):
                processed_str = date_str.replace('/', '-')
                if processed_str.count('-') == 1:
                    return f"{start_date.year}-{processed_str}"
                return processed_str
            df_new['날짜'] = pd.to_datetime(df_new['날짜'].apply(format_kamis_date))
            df_new['도매가격_원'] = pd.to_numeric(df_new['도매가격_원'].str.replace(',', ''), errors='coerce')
            return df_new
    except Exception as e:
        st.sidebar.error(f"KAMIS API 호출 중 오류: {e}")
    return pd.DataFrame()

# --- 5. Streamlit App UI and Main Logic ---
st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 분석 대시보드 (다중 AI 감성 & Naver 중심)")

bq_client = get_bq_connection()
models = load_models()
if bq_client is None:
    st.error("GCP 초기화에 실패했습니다. 앱을 재시작해주세요.")
    st.stop()
ensure_news_table_exists(bq_client)

# 세션 상태 초기화
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = []
if 'search_data' not in st.session_state:
    st.session_state.search_data = pd.DataFrame()
if 'news_data' not in st.session_state:
    st.session_state.news_data = pd.DataFrame()

# --- Sidebar UI ---
st.sidebar.header("⚙️ 분석 설정")
st.sidebar.subheader("1. 데이터 소스 선택")
data_src = st.sidebar.radio("데이터 소스", ["BigQuery", "CSV 업로드"])

if data_src == 'CSV 업로드':
    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드 (BigQuery와 동일 헤더)", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            st.session_state.raw_trade_df = df
            st.session_state.data_loaded = True
            st.session_state.selected_categories = sorted(df['Category'].astype(str).unique().tolist()) if 'Category' in df.columns else []
        except Exception as e:
            st.sidebar.error(f"파일을 읽는 중 오류 발생: {e}")
else:
    categories = get_categories_from_bq(bq_client)
    if not categories:
        st.sidebar.warning("BigQuery에 분석할 데이터가 없습니다.")
    else:
        selected_categories = st.sidebar.multiselect(
            "분석할 품목 선택", categories, default=st.session_state.get('selected_categories', [])
        )
        if st.sidebar.button("🚀 선택 품목 데이터 불러오기", key="load_trade"):
            if not selected_categories:
                st.sidebar.warning("카테고리를 선택해주세요.")
            else:
                df = get_trade_data_from_bq(bq_client, selected_categories)
                if df is not None and not df.empty:
                    st.session_state.raw_trade_df = df
                    st.session_state.data_loaded = True
                    st.session_state.selected_categories = selected_categories
                    st.rerun()
                else:
                    st.sidebar.error("데이터를 불러오지 못했습니다.")

if not st.session_state.data_loaded:
    st.info("👈 사이드바에서 분석할 데이터를 선택·불러오세요.")
    st.stop()

raw_trade_df = st.session_state.raw_trade_df
st.sidebar.success(f"**{', '.join(st.session_state.selected_categories)}** 데이터 로드 완료!")
st.sidebar.markdown("---")

st.sidebar.subheader("2. 분석 기간 및 키워드 설정")
file_start_date = pd.to_datetime(raw_trade_df['Date'].min())
file_end_date = pd.to_datetime(raw_trade_df['Date'].max())
start_date = pd.to_datetime(st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date))
end_date = pd.to_datetime(st.sidebar.date_input('종료일', file_end_date, min_value=start_date, max_value=file_end_date))

default_keywords = ", ".join(st.session_state.selected_categories) if st.session_state.selected_categories else ""
keyword_input = st.sidebar.text_input("트렌드/뉴스 분석 키워드", default_keywords)
search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]

st.sidebar.subheader("3. 외부 데이터 연동")
with st.sidebar.expander("🔑 API 키 입력"):
    kamis_api_key = st.text_input("KAMIS API Key", type="password")
    kamis_api_id = st.text_input("KAMIS API ID", type="password")
    naver_client_id = st.text_input("Naver API Client ID", type="password")
    naver_client_secret = st.text_input("Naver API Client Secret", type="password")

st.sidebar.markdown("##### KAMIS 농산물 가격")
kamis_item_name = st.sidebar.selectbox("품목 선택", list(KAMIS_FULL_DATA.keys()))
if kamis_item_name:
    kamis_kind_name = st.sidebar.selectbox("품종 선택", list(KAMIS_FULL_DATA[kamis_item_name]['kinds'].keys()))
    if st.sidebar.button("🌾 KAMIS 데이터 가져오기"):
        if kamis_api_key and kamis_api_id:
            item_info = KAMIS_FULL_DATA[kamis_item_name].copy()
            item_info['kind_code'] = item_info['kinds'][kamis_kind_name]
            item_info['rank_code'] = '01'
            st.session_state.wholesale_data = fetch_kamis_data(
                bq_client, item_info, start_date, end_date, {'key': kamis_api_key, 'id': kamis_api_id}
            )
        else:
            st.sidebar.error("KAMIS API Key와 ID를 모두 입력해주세요.")

st.sidebar.markdown("##### 네이버 트렌드 데이터 (검색/쇼핑)")
if st.sidebar.button("📈 네이버 트렌드 가져오기"):
    if search_keywords and naver_client_id and naver_client_secret:
        st.session_state.search_data = fetch_naver_trends_data(
            bq_client, search_keywords, start_date, end_date,
            {'id': naver_client_id, 'secret': naver_client_secret}
        )
    elif not search_keywords:
        st.sidebar.warning("트렌드 분석 키워드를 입력해주세요.")
    else:
        st.sidebar.error("Naver API 키를 입력해주세요.")

st.sidebar.markdown("##### 뉴스 감성 분석 (다중 모델)")
if st.sidebar.button("📰 뉴스 감성 분석 실행"):
    if search_keywords:
        st.session_state.news_data = get_news_with_multi_model_analysis(bq_client, models, search_keywords[0])
    else:
        st.sidebar.warning("뉴스 분석 키워드를 입력해주세요.")

# --- Main Display Tabs ---
raw_wholesale_df = st.session_state.get('wholesale_data', pd.DataFrame())
raw_search_df = st.session_state.get('search_data', pd.DataFrame())
raw_news_df = st.session_state.get('news_data', pd.DataFrame())

# (NEW) 메인 화면 '해제/재로딩' 빠른 액션바
st.markdown("### ⚡ 빠른 작업")
colA, colB, colC = st.columns(3)
with colA:
    if st.button("🧹 네이버 데이터 해제(초기화)"):
        st.session_state.search_data = pd.DataFrame()
        st.success("네이버 데이터셋을 해제했습니다. (메인화면 표시도 즉시 초기화)")
with colB:
    if st.button("🔁 네이버 데이터 다시 불러오기"):
        if search_keywords and naver_client_id and naver_client_secret:
            st.session_state.search_data = fetch_naver_trends_data(
                bq_client, search_keywords, start_date, end_date,
                {'id': naver_client_id, 'secret': naver_client_secret}
            )
            st.success("네이버 데이터셋을 현재 키워드로 재로딩했습니다.")
        else:
            st.warning("키워드 및 Naver API 키를 확인하세요.")
with colC:
    if st.button("📰 뉴스 분석 다시 실행"):
        if search_keywords:
            st.session_state.news_data = get_news_with_multi_model_analysis(bq_client, models, search_keywords[0])
            st.success("뉴스 감성 분석을 다시 실행했습니다.")
        else:
            st.warning("뉴스 분석 키워드를 입력하세요.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 원본 데이터",
    "2️⃣ 데이터 표준화",
    "3️⃣ 뉴스 감성 분석(다중 모델)",
    "4️⃣ 상관관계 분석",
    "📈 시계열 예측"
])

with tab1:
    st.subheader("A. 수출입 데이터 (선택 기간)")
    st.dataframe(raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)])
    st.subheader("B. 외부 가격 데이터")
    st.dataframe(raw_wholesale_df)
    st.subheader("C. 네이버 트렌드 데이터 (검색/쇼핑)")
    st.dataframe(raw_search_df)
    st.subheader("D. 뉴스 데이터")
    st.dataframe(raw_news_df)

with tab2:
    st.header("데이터 표준화: 주별(Weekly) 데이터로 변환")
    trade_df_in_range = raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)]
    filtered_trade_df = trade_df_in_range[trade_df_in_range['Category'].isin(st.session_state.selected_categories)].copy()

    if not filtered_trade_df.empty:
        filtered_trade_df.set_index('Date', inplace=True)
        trade_weekly = filtered_trade_df.resample('W-Mon').agg(
            수입액_USD=('Value', 'sum'),
            수입량_KG=('Volume', 'sum')
        ).copy()
        trade_weekly['수입단가_USD_KG'] = trade_weekly['수입액_USD'] / trade_weekly['수입량_KG']
        trade_weekly.index.name = '날짜'

        dfs_to_process = {'wholesale': raw_wholesale_df, 'search': raw_search_df, 'news': raw_news_df}
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
    st.header("뉴스 감성 분석 결과 (세 모델 교차 비교)")
    raw_news_df = st.session_state.get('news_data', pd.DataFrame())

    # (NEW) AI 뉴스 트렌드 요약 카드
    def _simple_tokenize_ko(text):
        # 형태소기 없어도 동작하는 가벼운 토크나이저 (영숫자/한글 단어)
        import re
        toks = re.findall(r"[가-힣A-Za-z0-9]+", text)
        stop = set(["한국", "정부", "시장", "경제", "가격", "관련", "등", "및", "뉴스", "기사", "최근", "전망"])
        return [t for t in toks if len(t) > 1 and t not in stop]

    if not raw_news_df.empty:
        dfn = raw_news_df.copy()
        dfn['날짜'] = pd.to_datetime(dfn['날짜'])

        # (NEW) 세 모델 평균/개별 시계열
        for col in ["FinBERT_Sentiment", "KLUE_Sentiment", "NSMC_Sentiment"]:
            if col not in dfn.columns:
                dfn[col] = np.nan
        dfn["AvgSentiment_3Models"] = dfn[["FinBERT_Sentiment", "KLUE_Sentiment", "NSMC_Sentiment"]].mean(axis=1, skipna=True)

        # 요약 기간 (최근 N일)
        N_days = 14
        cutoff = dfn['날짜'].max() - pd.Timedelta(days=N_days-1)
        df_recent = dfn[dfn['날짜'] >= cutoff]

        # 긍/부정 키워드(제목 기반 라이트 추출)
        pos_mask = dfn['AvgSentiment_3Models'] > 0.15
        neg_mask = dfn['AvgSentiment_3Models'] < -0.15
        pos_words = pd.Series(sum([_simple_tokenize_ko(t) for t in dfn.loc[pos_mask, 'Title'].dropna().tolist()], [])).value_counts().head(5)
        neg_words = pd.Series(sum([_simple_tokenize_ko(t) for t in dfn.loc[neg_mask, 'Title'].dropna().tolist()], [])).value_counts().head(5)

        avg_recent = round(df_recent['AvgSentiment_3Models'].mean(), 3) if not df_recent.empty else np.nan
        trend_line = "긍정적" if avg_recent > 0.05 else ("부정적" if avg_recent < -0.05 else "중립적")
        pos_kw = ", ".join(pos_words.index.tolist())
        neg_kw = ", ".join(neg_words.index.tolist())

        with st.container():
            st.markdown("#### 🤖 AI 뉴스 트렌드 요약")
            st.info(
                f"최근 {N_days}일 기준 평균 종합 감성은 **{avg_recent if not np.isnan(avg_recent) else 'N/A'}**로 **{trend_line}**입니다. "
                f"긍정 키워드: {pos_kw if pos_kw else '없음'} / 부정 키워드: {neg_kw if neg_kw else '없음'}"
            )

        # 주별 집계 (세 모델)
        news_weekly = dfn.dropna(subset=['날짜']).set_index('날짜').resample('W-Mon').mean(numeric_only=True)
        news_weekly.index.name = '날짜'

        # 시계열: 3모델 비교
        fig_multi = px.line(
            news_weekly.reset_index(),
            x='날짜',
            y=['FinBERT_Sentiment', 'KLUE_Sentiment', 'NSMC_Sentiment', 'AvgSentiment_3Models'],
            title="주별 평균 뉴스 감성 점수 (FinBERT / KLUE / NSMC / 평균)"
        )
        fig_multi.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig_multi, use_container_width=True)

        # 키워드별 라인 (FinBERT 기준, 다중키워드면 색상 분리)
        if 'Keyword' in dfn.columns and dfn['Keyword'].nunique() > 1:
            fig_kw = px.line(dfn.sort_values('날짜'), x='날짜', y='FinBERT_Sentiment', color='Keyword', title="키워드별 FinBERT 감성 추이")
            st.plotly_chart(fig_kw, use_container_width=True)

        st.markdown("---")
        st.subheader("수집 뉴스 원본 및 감성 점수 (최신순)")
        show_cols = ['날짜', 'Title', 'Keyword', 'FinBERT_Sentiment', 'KLUE_Sentiment', 'NSMC_Sentiment',
                     'FinBERT_Label', 'KLUE_Label', 'NSMC_Label', 'RawUrl']
        st.dataframe(dfn.sort_values(by='날짜', ascending=False)[[c for c in show_cols if c in dfn.columns]])

    else:
        st.info("사이드바 또는 상단 빠른 작업에서 뉴스 감성 분석을 실행해주세요.")

# (tab4 - 상관관계)
with tab4:
    st.header("상관관계 분석")
    if 'weekly_dfs' in st.session_state:
        weekly_dfs = st.session_state['weekly_dfs']
        dfs_to_concat = [df for df in weekly_dfs.values() if not df.empty]
        if len(dfs_to_concat) > 1:
            final_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), dfs_to_concat)
            final_df = final_df.interpolate(method='linear', limit_direction='both').dropna(how='all', axis=1).dropna()
            st.session_state['final_df'] = final_df

            st.subheader("통합 데이터 시각화")
            df_long = final_df.reset_index().melt(id_vars='날짜', var_name='데이터 종류', value_name='값')
            fig = px.line(df_long, x='날짜', y='값', color='데이터 종류', title="통합 데이터 시계열 추이")
            st.plotly_chart(fig, use_container_width=True)

            if len(final_df.columns) > 1:
                st.subheader("상관관계 히트맵")
                corr_matrix = final_df.corr(numeric_only=True)
                fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1])
                st.plotly_chart(fig_heatmap, use_container_width=True)

                st.subheader("산점도 행렬 (Scatter Matrix)")
                fig_matrix = px.scatter_matrix(final_df.reset_index(),
                                               dimensions=final_df.columns,
                                               title="통합 데이터 산점도 행렬")
                st.plotly_chart(fig_matrix, use_container_width=True)
        else:
            st.warning("상관관계 분석을 위해 둘 이상의 데이터가 필요합니다.")
    else:
        st.warning("2단계에서 데이터가 처리되지 않았습니다.")

with tab5:
    st.header("시계열 분해 및 예측 (by Prophet)")
    if 'final_df' in st.session_state and not st.session_state['final_df'].empty:
        final_df = st.session_state['final_df']
        forecast_col = st.selectbox("예측 대상 변수 선택", final_df.columns)
        if st.button("📈 선택한 변수로 예측 실행하기"):
            ts_data = final_df[[forecast_col]].dropna()
            if len(ts_data) < 24:
                st.warning(f"최소 24주 이상의 데이터가 필요합니다. 현재: {len(ts_data)}주")
            else:
                with st.spinner(f"'{forecast_col}' 예측 모델 학습 중..."):
                    st.subheader(f"'{forecast_col}' 시계열 분해")
                    period = 52 if len(ts_data) >= 104 else max(4, int(len(ts_data) / 2))
                    decomposition = seasonal_decompose(ts_data[forecast_col], model='additive', period=period)
                    fig_decompose = go.Figure()
                    fig_decompose.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed'))
                    fig_decompose.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'))
                    fig_decompose.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'))
                    st.session_state['fig_decompose'] = fig_decompose

                    st.subheader(f"'{forecast_col}' 미래 12주 예측")
                    prophet_df = ts_data.reset_index().rename(columns={'날짜': 'ds', forecast_col: 'y'})
                    m = Prophet()
                    m.fit(prophet_df)
                    future = m.make_future_dataframe(periods=12, freq='W')
                    forecast = m.predict(future)
                    fig_forecast = plot_plotly(m, forecast)
                    st.session_state['fig_forecast'] = fig_forecast
                    st.session_state['forecast_data'] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)

        if 'fig_decompose' in st.session_state:
            st.plotly_chart(st.session_state['fig_decompose'], use_container_width=True)
        if 'fig_forecast' in st.session_state:
            st.plotly_chart(st.session_state['fig_forecast'], use_container_width=True)
            st.dataframe(st.session_state['forecast_data'])
    else:
        st.info("4번 탭에서 데이터가 통합되어야 예측을 수행할 수 있습니다.")
