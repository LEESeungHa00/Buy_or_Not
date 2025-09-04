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
from google.cloud import bigquery, language_v1
import pandas_gbq
import feedparser
from urllib.parse import quote
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.plot import plot_plotly

# ==============================================================================
# --- 1. Constants and Configuration ---
# ==============================================================================
BQ_DATASET = "data_explorer"
BQ_TABLE_NAVER = "naver_trends_cache"
BQ_TABLE_NEWS = "news_sentiment_google_nlp" # 새 테이블 이름

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

# ==============================================================================
# --- 2. GCP Connection and Helper Functions ---
# ==============================================================================

@st.cache_resource
def get_bq_connection():
    """BigQuery에 연결하고 클라이언트 객체를 반환합니다."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        client = bigquery.Client(credentials=creds, project=creds.project_id)
        return client
    except Exception as e:
        st.error(f"Google BigQuery 연결 실패: secrets.toml을 확인하세요. 오류: {e}")
        return None

@st.cache_resource
def get_gcp_nlp_client():
    """Google Cloud Natural Language API 클라이언트를 반환합니다."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        nlp_client = language_v1.LanguageServiceClient(credentials=creds)
        return nlp_client
    except Exception as e:
        st.error(f"Google NLP 연결 실패: {e}")
        return None

def ensure_news_table_exists(client):
    """새로운 뉴스 분석 결과에 맞는 스키마로 BigQuery 테이블이 있는지 확인하고 없으면 생성합니다."""
    project_id = client.project
    full_table_id = f"{project_id}.{BQ_DATASET}.{BQ_TABLE_NEWS}"
    try:
        client.get_table(full_table_id)
    except Exception:
        st.write(f"뉴스 분석 테이블 '{full_table_id}'을 새로 생성합니다.")
        schema = [
            bigquery.SchemaField("날짜", "DATE"),
            bigquery.SchemaField("Title", "STRING"),
            bigquery.SchemaField("Keyword", "STRING"),
            bigquery.SchemaField("Sentiment", "FLOAT"),
            bigquery.SchemaField("Magnitude", "FLOAT"),
            bigquery.SchemaField("InsertedAt", "TIMESTAMP"),
        ]
        table = bigquery.Table(full_table_id, schema=schema)
        client.create_table(table)
        st.success(f"테이블 '{BQ_TABLE_NEWS}' 생성 완료.")

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
    except Exception as e:
        st.error(f"Naver API 오류 발생: {e}")
        return None

# ==============================================================================
# --- 3. Main Data Fetching Functions ---
# ==============================================================================

def get_trade_data_from_bq(client, categories):
    """선택된 카테고리에 대한 수출입 데이터를 BigQuery에서 가져옵니다."""
    # (이하 함수 내용은 이전과 동일)
    if not categories: return pd.DataFrame()
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
    """BigQuery 캐시를 활용하여 네이버 데이터랩 데이터를 긴 기간에 대해 가져옵니다."""
    # (이하 함수 내용은 이전과 동일)
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
            current_end = current_start + timedelta(days=89)
            if current_end > end_date:
                current_end = end_date
            
            # (이하 API 호출 로직은 이전과 동일)
            NAVER_SHOPPING_CAT_MAP = {'아보카도': "50000007", '바나나': "50000007", '사과': "50000007"}
            all_data_chunk = []
            for keyword in keywords:
                keyword_dfs = []
                body_search = json.dumps({"startDate": current_start.strftime('%Y-%m-%d'), "endDate": current_end.strftime('%Y-%m-%d'), "timeUnit": "date", "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]})
                search_res = call_naver_api("https://openapi.naver.com/v1/datalab/search", body_search, naver_keys)
                if search_res and search_res.get('results') and search_res['results'][0]['data']:
                    df_search = pd.DataFrame(search_res['results'][0]['data'])
                    if not df_search.empty: keyword_dfs.append(df_search.rename(columns={'period': '날짜', 'ratio': f'NaverSearch_{keyword}'}))
                
                if keyword.lower().replace(' ', '') in NAVER_SHOPPING_CAT_MAP:
                    category_id = NAVER_SHOPPING_CAT_MAP[keyword.lower().replace(' ', '')]
                    body_shop = json.dumps({"startDate": current_start.strftime('%Y-%m-%d'),"endDate": current_end.strftime('%Y-%m-%d'), "timeUnit": "date", "category": [{"name": keyword, "param": [category_id]}], "keyword": [{"name": keyword, "param": [keyword]}]})
                    shop_res = call_naver_api("https://openapi.naver.com/v1/datalab/shopping/categories", body_shop, naver_keys)
                    if shop_res and shop_res.get('results') and shop_res['results'][0]['data']:
                        df_shop = pd.DataFrame(shop_res['results'][0]['data'])
                        if not df_shop.empty: keyword_dfs.append(df_shop.rename(columns={'period': '날짜', 'ratio': f'NaverShop_{keyword}'}))
                
                if keyword_dfs:
                    for df in keyword_dfs: df['날짜'] = pd.to_datetime(df['날짜'])
                    merged_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), keyword_dfs)
                    all_data_chunk.append(merged_df)
            
            if all_data_chunk:
                new_data_list.append(reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_data_chunk))
            current_start = current_end + timedelta(days=1)
    else:
        st.sidebar.success("✔️ 네이버 트렌드: 모든 데이터가 캐시에 있습니다.")

    if new_data_list:
        non_empty_dfs = [df for df in new_data_list if not df.empty]
        if non_empty_dfs:
            df_new = pd.concat(non_empty_dfs, ignore_index=True)
            df_new['날짜'] = pd.to_datetime(df_new['날짜'])
            df_combined = pd.concat([df_cache, df_new], ignore_index=True).drop_duplicates(subset=['날짜'], keep='last').sort_values(by='날짜').reset_index(drop=True)
            pandas_gbq.to_gbq(df_combined, table_id, project_id=project_id, if_exists="replace", credentials=_client._credentials)
            df_final = df_combined
        else:
            df_final = df_cache
    else:
        df_final = df_cache

    if df_final.empty: return pd.DataFrame()
    return df_final[(df_final['날짜'] >= start_date) & (df_final['날짜'] <= end_date)].reset_index(drop=True)

def fetch_kamis_data(client, item_info, start_date, end_date, kamis_keys):
    """KAMIS에서 기간별 도매 가격 데이터를 가져옵니다."""
    # (이하 함수 내용은 이전과 동일)
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    item_code, kind_code = item_info['item_code'], item_info['kind_code']
    url = (f"http://www.kamis.or.kr/service/price/xml.do?action=periodWholesaleProductList"
           f"&p_product_cls_code=01&p_startday={start_str}&p_endday={end_str}"
           f"&p_item_category_code={item_info['cat_code']}&p_item_code={item_code}&p_kind_code={kind_code}"
           f"&p_product_rank_code={item_info['rank_code']}&p_convert_kg_yn=Y"
           f"&p_cert_key={kamis_keys['key']}&p_cert_id={kamis_keys['id']}&p_returntype=json")
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200 and "data" in response.json() and "item" in response.json()["data"]:
            price_data = response.json()["data"]["item"]
            if not price_data: return pd.DataFrame()
            df_new = pd.DataFrame(price_data)[['regday', 'price']].rename(columns={'regday': '날짜', 'price': '도매가격_원'})
            df_new['날짜'] = pd.to_datetime(df_new['날짜'])
            df_new['도매가격_원'] = pd.to_numeric(df_new['도매가격_원'].str.replace(',', ''), errors='coerce')
            return df_new
    except Exception as e:
        st.sidebar.error(f"KAMIS API 호출 중 오류: {e}")
    return pd.DataFrame()

def fetch_and_analyze_news_lightweight(_bq_client, _nlp_client, keyword, days_limit=7):
    """뉴스를 수집하고 Google NLP로 분석 후, BigQuery에 캐싱하는 경량화된 함수."""
    # (이하 함수 내용은 이전과 동일)
    project_id = _bq_client.project
    full_table_id = f"{project_id}.{BQ_DATASET}.{BQ_TABLE_NEWS}"

    try:
        time_limit = datetime.now(timezone.utc) - timedelta(days=days_limit)
        query = f"SELECT * FROM `{full_table_id}` WHERE Keyword = @keyword AND InsertedAt >= @time_limit ORDER BY 날짜 DESC"
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("keyword", "STRING", keyword), bigquery.ScalarQueryParameter("time_limit", "TIMESTAMP", time_limit)])
        df_cache = _bq_client.query(query, job_config=job_config).to_dataframe()
    except Exception:
        df_cache = pd.DataFrame()

    if not df_cache.empty:
        st.sidebar.success(f"✔️ '{keyword}' 최신 뉴스 결과를 캐시에서 로드했습니다.")
        return df_cache

    st.sidebar.warning(f"'{keyword}'에 대한 최신 캐시가 없습니다. 새로 분석합니다.")
    all_news = []
    rss_url = f"https://news.google.com/rss/search?q={quote(keyword)}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)
    
    for entry in feed.entries[:20]:
        title = entry.get('title', '')
        if not title: continue
        try:
            pub_date = pd.to_datetime(entry.get('published')).date()
            all_news.append({"날짜": pub_date, "Title": title})
        except: continue
    
    if not all_news: return pd.DataFrame()

    df_new = pd.DataFrame(all_news).drop_duplicates(subset=["Title"])
    
    def analyze_sentiment_with_google(text_content, nlp_client):
        if not text_content or not nlp_client: return 0.0, 0.0
        document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_sentiment(request={'document': document})
        return response.document_sentiment.score, response.document_sentiment.magnitude

    sentiments = [analyze_sentiment_with_google(title, _nlp_client) for title in df_new['Title']]
    df_new['Sentiment'] = [s[0] for s in sentiments]
    df_new['Magnitude'] = [s[1] for s in sentiments]
    df_new['Keyword'] = keyword
    df_new['InsertedAt'] = datetime.now(timezone.utc)
    
    if not df_new.empty:
        df_to_gbq = df_new[["날짜", "Title", "Keyword", "Sentiment", "Magnitude", "InsertedAt"]]
        pandas_gbq.to_gbq(df_to_gbq, full_table_id, project_id=project_id, if_exists="append", credentials=_bq_client._credentials)
    
    return df_to_gbq

# ==============================================================================
# --- 4. Streamlit App Main Logic ---
# ==============================================================================

st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 분석 대시보드")

# --- Initialize GCP Clients ---
bq_client = get_bq_connection()
nlp_client = get_gcp_nlp_client()
if bq_client is None or nlp_client is None:
    st.stop()

# --- Initialize News Table in BigQuery ---
ensure_news_table_exists(bq_client)

# --- Initialize Session State ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- Sidebar UI ---
st.sidebar.header("⚙️ 분석 설정")

# 1. Select Main Data
st.sidebar.subheader("1. 분석 대상 품목 선택")
categories = get_categories_from_bq(bq_client)
if not categories:
    st.sidebar.warning("BigQuery에 분석할 데이터가 없습니다.")
else:
    selected_categories = st.sidebar.multiselect(
        "분석할 품목 카테고리 선택", categories, default=st.session_state.get('selected_categories', [])
    )
    if st.sidebar.button("🚀 선택 품목 데이터 불러오기"):
        if not selected_categories:
            st.sidebar.warning("카테고리를 선택해주세요.")
        else:
            df = get_trade_data_from_bq(bq_client, selected_categories)
            if df is not None and not df.empty:
                st.session_state.raw_trade_df = df
                st.session_state.data_loaded = True
                st.session_state.selected_categories = selected_categories
                st.rerun() # Reload to update the main page
            else:
                st.sidebar.error("데이터를 불러오지 못했습니다.")

# Stop if main data is not loaded yet
if not st.session_state.data_loaded:
    st.info("👈 사이드바에서 분석할 카테고리를 선택하고 버튼을 눌러주세요.")
    st.stop()

# --- Main App Logic continues only if data is loaded ---
raw_trade_df = st.session_state.raw_trade_df
st.sidebar.success(f"**{', '.join(st.session_state.selected_categories)}** 데이터 로드 완료!")
st.sidebar.markdown("---")

# 2. Set Date Range & Keywords
st.sidebar.subheader("2. 분석 기간 및 키워드 설정")
file_start_date = raw_trade_df['Date'].min()
file_end_date = raw_trade_df['Date'].max()
start_date = pd.to_datetime(st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date))
end_date = pd.to_datetime(st.sidebar.date_input('종료일', file_end_date, min_value=start_date, max_value=file_end_date))
default_keywords = ", ".join(st.session_state.selected_categories)
keyword_input = st.sidebar.text_input("트렌드/뉴스 분석 키워드", default_keywords)
search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]

# 3. Fetch External Data
st.sidebar.subheader("3. 외부 데이터 연동")
with st.sidebar.expander("🔑 API 키 입력"):
    kamis_api_key = st.text_input("KAMIS API Key", type="password")
    kamis_api_id = st.text_input("KAMIS API ID", type="password")
    naver_client_id = st.text_input("Naver API Client ID", type="password")
    naver_client_secret = st.text_input("Naver API Client Secret", type="password")

if st.sidebar.button("🔗 모든 외부 데이터 가져오기"):
    # Fetch KAMIS Data
    with st.spinner("KAMIS 데이터 가져오는 중..."):
        # For simplicity, we fetch for the first selected category if it's in KAMIS_FULL_DATA
        kamis_item_name = next((cat for cat in st.session_state.selected_categories if cat in KAMIS_FULL_DATA), None)
        if kamis_item_name and kamis_api_key and kamis_api_id:
            item_info = KAMIS_FULL_DATA[kamis_item_name]
            item_info['item_code'] = item_info['item_code']
            item_info['kind_code'] = list(item_info['kinds'].values())[0] # Default to first kind
            item_info['rank_code'] = '01'
            st.session_state.wholesale_data = fetch_kamis_data(bq_client, item_info, start_date, end_date, {'key': kamis_api_key, 'id': kamis_api_id})
        elif not kamis_item_name:
            st.sidebar.info("선택된 품목에 대한 KAMIS 정보가 없습니다.")
        else:
            st.sidebar.warning("KAMIS API 키와 ID를 입력해주세요.")

    # Fetch Naver Trend Data
    with st.spinner("네이버 트렌드 데이터 가져오는 중..."):
        if search_keywords and naver_client_id and naver_client_secret:
            st.session_state.search_data = fetch_naver_trends_data(bq_client, search_keywords, start_date, end_date, {'id': naver_client_id, 'secret': naver_client_secret})
        elif not search_keywords:
            st.sidebar.warning("트렌드 분석 키워드를 입력해주세요.")
        else:
            st.sidebar.warning("Naver API 키를 입력해주세요.")
            
    # Fetch News Sentiment Data
    with st.spinner("뉴스 감성 데이터 가져오는 중..."):
        if search_keywords:
            # For simplicity, fetch for the first keyword
            st.session_state.news_data = fetch_and_analyze_news_lightweight(bq_client, nlp_client, search_keywords[0])
        else:
            st.sidebar.warning("뉴스 분석 키워드를 입력해주세요.")

# --- Main Display Tabs ---
raw_wholesale_df = st.session_state.get('wholesale_data', pd.DataFrame())
raw_search_df = st.session_state.get('search_data', pd.DataFrame())
raw_news_df = st.session_state.get('news_data', pd.DataFrame())

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1️⃣ 원본 데이터", "2️⃣ 데이터 표준화", "3️⃣ 뉴스 감성 분석", "4️⃣ 상관관계 분석", "📈 시계열 예측"])

with tab1:
    st.subheader("A. 수출입 데이터 (선택 기간)")
    st.dataframe(raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)])
    st.subheader("B. 외부 가격 데이터"); st.dataframe(raw_wholesale_df)
    st.subheader("C. 트렌드 데이터"); st.dataframe(raw_search_df)
    st.subheader("D. 뉴스 데이터"); st.dataframe(raw_news_df)
    
with tab2:
    st.header("데이터 표준화: 주별(Weekly) 데이터로 변환")
    # (이하 탭 내용은 이전과 동일)
    trade_df_in_range = raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)]
    filtered_trade_df = trade_df_in_range[trade_df_in_range['Category'].isin(st.session_state.selected_categories)].copy()
    
    if not filtered_trade_df.empty:
        filtered_trade_df.set_index('Date', inplace=True)
        trade_weekly = filtered_trade_df.resample('W-Mon').agg(수입액_USD=('Value', 'sum'), 수입량_KG=('Volume', 'sum')).copy()
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
        st.info("사이드바에서 외부 데이터를 가져와주세요.")

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
                # (이하 예측 로직은 이전과 동일)
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
