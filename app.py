import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

# --- BigQuery Connection (Manual Method for Stability) ---
@st.cache_resource
def get_bq_connection():
    """BigQuery에 직접 연결하고 클라이언트 객체를 반환합니다."""
    try:
        # st.secrets에서 직접 인증 정보 가져오기
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        # BigQuery 클라이언트 생성
        client = bigquery.Client(credentials=creds, project=creds.project_id)
        return client
    except Exception as e:
        st.error(f"Google BigQuery 연결에 실패했습니다. secrets.toml의 [gcp_service_account] 설정을 확인하세요: {e}")
        return None

# --- Data Fetching & Processing Functions (BigQuery Version) ---
def get_trade_data_from_bq(client):
    """BigQuery의 tds_data 테이블에서 데이터를 로드합니다."""
    project_id = client.project
    table_id = f"{project_id}.data_explorer.tds_data"
    try:
        count_query = f"SELECT count(*) FROM `{table_id}`"
        count_result = client.query(count_query).to_dataframe()
        total_rows = count_result.iloc[0,0]
        
        with st.spinner(f"BigQuery에서 {total_rows:,}개의 행을 로드하는 중..."):
            sql = f"SELECT * FROM `{table_id}`"
            df = client.query(sql).to_dataframe()

        # 데이터 타입 변환
        for col in df.columns:
            if 'price' in col.lower() or 'value' in col.lower() or 'volume' in col.lower():
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"BigQuery에서 TDS 데이터를 읽는 중 오류 발생: {e}")
        return pd.DataFrame()

def add_trade_data_to_bq(client, df):
    """새로운 데이터를 BigQuery 테이블에 추가합니다."""
    project_id = client.project
    table_id = f"{project_id}.data_explorer.tds_data"
    try:
        # BigQuery 규칙에 맞게 컬럼 이름 변경
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)
        with st.spinner("새 데이터를 BigQuery에 저장하는 중..."):
            # pandas_gbq를 사용하여 데이터 추가
            pandas_gbq.to_gbq(df, table_id, project_id=project_id, if_exists="append", credentials=client._credentials)
        st.success("새 데이터가 BigQuery에 성공적으로 저장되었습니다.")
    except Exception as e:
        st.error(f"BigQuery에 데이터를 저장하는 중 오류 발생: {e}")

# (기타 API fetch 함수들은 기존과 동일하게 유지)
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

# --- Constants ---
COFFEE_TICKERS_YFINANCE = {"미국 커피 C": "KC=F", "런던 로부스타": "RC=F"}
KAMIS_CATEGORIES = {"채소류": "100", "과일류": "200", "축산물": "300", "수산물": "400"}
KAMIS_ITEMS = {"채소류": {"배추": "111", "무": "112", "양파": "114", "마늘": "141"}, "과일류": {"사과": "211", "바나나": "214", "아보카도": "215"}, "축산물": {"소고기": "311", "돼지고기": "312"}, "수산물": {"고등어": "411", "오징어": "413"}}

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 대시보드 (Google BigQuery 연동)")

bq_client = get_bq_connection()
if bq_client is None: st.stop()

st.sidebar.header("⚙️ 분석 설정")

# --- App Startup Workflow (BigQuery Version) ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    if st.sidebar.button("🚀 데이터 분석 시작하기 (BigQuery에서 불러오기)"):
        st.session_state.raw_trade_df = get_trade_data_from_bq(bq_client)
        if st.session_state.raw_trade_df is not None and not st.session_state.raw_trade_df.empty:
            st.session_state.data_loaded = True
            st.rerun()
        else:
            st.sidebar.error("BigQuery에서 데이터를 불러오지 못했습니다. 테이블이 비어있거나 접근 권한을 확인하세요.")
    else:
        st.info("👈 사이드바의 '데이터 분석 시작하기' 버튼을 눌러주세요.")
        st.sidebar.subheader("또는, 새 수출입 데이터 추가")
        uploaded_file = st.sidebar.file_uploader("새 파일 업로드하여 BigQuery에 추가", type=['csv', 'xlsx'])
        if uploaded_file:
            if st.sidebar.button("업로드 파일 BigQuery에 저장"):
                df_new = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                add_trade_data_to_bq(bq_client, df_new)
                st.session_state.data_loaded = False # Reset state to force reload
                st.rerun()
        st.stop()

# --- Main App Logic (runs only after data is loaded) ---
raw_trade_df = st.session_state.raw_trade_df

st.sidebar.subheader("2. 분석 대상 설정")
try:
    file_start_date, file_end_date = raw_trade_df['Date'].min(), raw_trade_df['Date'].max()
    category_options = sorted(raw_trade_df['Category'].astype(str).unique())
    selected_categories = st.sidebar.multiselect("분석할 품목 카테고리 선택", category_options, default=category_options[0] if category_options else None)
    keyword_input = st.sidebar.text_input("3. 검색어 입력 (쉼표로 구분)", ", ".join(selected_categories) if selected_categories else None)
    search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
    st.sidebar.subheader("4. 분석 기간 설정")
    start_date_input = st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date)
    end_date_input = st.sidebar.date_input('종료일', file_end_date, min_value=start_date_input, max_value=file_end_date)
    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)
except Exception as e:
    st.error(f"데이터 처리 중 오류가 발생했습니다. 'Date' 또는 'Category' 컬럼의 데이터 형식을 확인해주세요: {e}")
    st.stop()

# --- External Data Loading Section ---
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

# --- Main Display Area ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 원본 데이터 확인", "2️⃣ 데이터 표준화", "3️⃣ 최종 통합 데이터"])
with tab1:
    st.subheader("A. 수출입 데이터 (from BigQuery)"); 
    st.dataframe(raw_trade_df.head())
    st.subheader("B. 외부 가격 데이터"); st.dataframe(raw_wholesale_df.head())
    st.subheader("C. 검색량 데이터"); st.dataframe(raw_search_df.head())

with tab2:
    if not selected_categories: st.warning("분석할 카테고리를 선택해주세요.")
    else:
        st.subheader("2-1. 분석 대상 품목 필터링")
        trade_df_in_range = raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)]
        filtered_trade_df = trade_df_in_range[trade_df_in_range['Category'].isin(selected_categories)].copy()
        st.write(f"선택된 카테고리: **{', '.join(selected_categories)}**"); st.dataframe(filtered_trade_df.head())
        
        st.subheader("2-2. 주(Week) 단위 데이터로 집계")
        if not filtered_trade_df.empty:
            filtered_trade_df.set_index('Date', inplace=True)
            value_col = 'Value' if 'Value' in filtered_trade_df.columns else 'Value'
            volume_col = 'Volume' if 'Volume' in filtered_trade_df.columns else 'Volume'
            
            trade_weekly = filtered_trade_df.resample('W-Mon').agg({value_col: 'sum', volume_col: 'sum'})
            trade_weekly['수입단가_USD_KG'] = trade_weekly[value_col] / trade_weekly[volume_col]
            trade_weekly.columns = ['수입액_USD', '수입량_KG', '수입단가_USD_KG']
            
            wholesale_weekly = pd.DataFrame()
            if not raw_wholesale_df.empty:
                date_col = '조사일자' if '조사일자' in raw_wholesale_df.columns else '날짜'
                raw_wholesale_df[date_col] = pd.to_datetime(raw_wholesale_df[date_col], errors='coerce')
                wholesale_df_processed = raw_wholesale_df.set_index(date_col)
                price_cols = [col for col in wholesale_df_processed.columns if '가격' in col]
                agg_dict = {col: 'mean' for col in price_cols}
                if agg_dict:
                    wholesale_weekly = wholesale_df_processed.resample('W-Mon').agg(agg_dict)
                    if '도매가격_원' in wholesale_weekly.columns:
                        wholesale_weekly['도매가격_USD'] = wholesale_weekly['도매가격_원'] / 1350
                        wholesale_weekly.drop(columns=['도매가격_원'], inplace=True)
            
            search_weekly = pd.DataFrame()
            if not raw_search_df.empty:
                raw_search_df['날짜'] = pd.to_datetime(raw_search_df['날짜'], errors='coerce')
                search_df_processed = raw_search_df.set_index('날짜')
                numeric_cols = search_df_processed.select_dtypes(include=np.number).columns
                search_weekly = search_df_processed.resample('W-Mon').agg({col: 'mean' for col in numeric_cols})
            
            st.write("▼ 일별(Daily) vs 주별(Weekly) 수입량 비교")
            col1, col2 = st.columns(2)
            with col1: st.line_chart(filtered_trade_df[volume_col])
            with col2: st.line_chart(trade_weekly['수입량_KG'])
        else: st.warning("선택된 카테고리에 해당하는 데이터가 없습니다.")

with tab3:
    if 'trade_weekly' in locals() and not trade_weekly.empty:
        dfs_to_concat = [trade_weekly]
        if 'wholesale_weekly' in locals() and not wholesale_weekly.empty: dfs_to_concat.append(wholesale_weekly)
        if 'search_weekly' in locals() and not search_weekly.empty: dfs_to_concat.append(search_weekly)
        
        final_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs_to_concat)
        final_df = final_df.interpolate(method='linear', limit_direction='forward').dropna(how='all')
        
        st.dataframe(final_df)
        st.subheader("최종 통합 데이터 시각화")
        if not final_df.empty:
            fig = px.line(final_df, labels={'value': '값', 'index': '날짜', 'variable': '데이터 종류'}, title="최종 통합 데이터 시계열 추이")
            st.plotly_chart(fig, use_container_width=True)
    else: st.warning("통합할 데이터가 없습니다.")

