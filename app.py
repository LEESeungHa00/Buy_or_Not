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
import gspread
from google.oauth2.service_account import Credentials
import time

# --- Google Sheets Connection ---
@st.cache_resource
def get_gspread_client():
    """Google Sheets 클라이언트를 초기화하고 반환합니다."""
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Google Sheets 인증에 실패했습니다. st.secrets 설정을 확인하세요: {e}")
        return None

@st.cache_data(ttl=600)
def read_gsheet(_gs_client, sheet_name):
    """지정된 시트에서 모든 데이터를 읽어 DataFrame으로 반환합니다."""
    try:
        spreadsheet = _gs_client.open_by_key(st.secrets["google_sheet_key"])
        worksheet = spreadsheet.worksheet(sheet_name)
        return pd.DataFrame(worksheet.get_all_records())
    except gspread.exceptions.WorksheetNotFound:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"'{sheet_name}' 시트를 읽는 중 오류 발생: {e}")
        return pd.DataFrame()

def update_gsheet(client, sheet_name, df_to_add):
    """지정된 시트에 새로운 데이터를 추가합니다. 헤더는 필요 시 한 번만 기록합니다."""
    if df_to_add.empty: return
    try:
        spreadsheet = client.open_by_key(st.secrets["google_sheet_key"])
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
            # 시트가 비어있으면 헤더 추가
            if not worksheet.get_all_records():
                 worksheet.update([df_to_add.columns.values.tolist()])
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=1, cols=len(df_to_add.columns))
            worksheet.update([df_to_add.columns.values.tolist()])

        worksheet.append_rows(df_to_add.astype(str).values.tolist(), value_input_option='USER_ENTERED')
    except Exception as e:
        st.error(f"'{sheet_name}' 시트 업데이트 중 오류 발생: {e}")

# --- Data Fetching & Processing Functions ---
def process_new_trade_data(gs_client, uploaded_file):
    """새로 업로드된 파일을 처리하고 Google Sheet에 저장합니다."""
    sheet_name = "TDS"
    file_name = uploaded_file.name
    try:
        df = pd.read_csv(uploaded_file) if file_name.endswith('.csv') else pd.read_excel(uploaded_file)
        if 'Date' not in df.columns or 'Category' not in df.columns:
            st.error("업로드된 파일에 'Date'와 'Category' 컬럼이 모두 필요합니다."); return None

        df['Date'] = pd.to_datetime(df['Date'])
        df_to_save = df.copy()
        df_to_save['file_name'] = file_name
        
        with st.spinner(f"'{file_name}'을 Google Sheet에 저장 중..."):
            update_gsheet(gs_client, sheet_name, df_to_save)
        st.sidebar.info(f"'{file_name}'을 Google Sheet에 저장했습니다.")
        return df
    except Exception as e:
        st.error(f"파일 처리 중 오류: {e}"); return None

def fetch_yfinance_data(gs_client, tickers, start_date, end_date):
    sheet_name = "yfinance"
    all_data = []
    df_sheet = read_gsheet(gs_client, sheet_name)
    if not df_sheet.empty: df_sheet['조사일자'] = pd.to_datetime(df_sheet['조사일자'])

    for name, ticker in tickers.items():
        if not df_sheet.empty and 'Ticker' in df_sheet.columns:
            cached_data = df_sheet[(df_sheet['Ticker'] == ticker) & (df_sheet['조사일자'] >= pd.to_datetime(start_date)) & (df_sheet['조사일자'] <= pd.to_datetime(end_date))]
            if len(cached_data) >= (end_date - start_date).days * 0.9: # 데이터가 충분하면 캐시 사용
                 st.sidebar.info(f"'{name}' 데이터를 Google Sheet에서 로드했습니다.")
                 all_data.append(cached_data.rename(columns={'Price': f'{name} 선물가격(USD)'})[['조사일자', f'{name} 선물가격(USD)']])
                 continue

        with st.spinner(f"'{name}' 데이터를 Yahoo Finance API에서 가져오는 중..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                df = data[['Close']].copy().reset_index().rename(columns={'Date': '조사일자', 'Close': 'Price'})
                df['Ticker'] = ticker
                update_gsheet(gs_client, sheet_name, df)
                st.sidebar.info(f"'{name}' 데이터를 API에서 가져와 Google Sheet에 저장했습니다.")
                all_data.append(df.rename(columns={'Price': f'{name} 선물가격(USD)'})[['조사일자', f'{name} 선물가격(USD)']])
            else: st.sidebar.warning(f"'{name}'에 대한 API 데이터가 없습니다.")

    if not all_data: return pd.DataFrame()
    return reduce(lambda left, right: pd.merge(left, right, on='조사일자', how='outer'), all_data)

def fetch_trends_data(gs_client, keywords, start_date, end_date, naver_keys):
    sheet_name = "네이버데이터랩"
    all_data = []
    
    for keyword in keywords:
        with st.spinner(f"'{keyword}' 검색량 데이터를 API에서 가져오는 중..."):
            pytrends = TrendReq(hl='ko-KR', tz=540)
            timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='KR', gprop='')
            google_df = pytrends.interest_over_time()
            if not google_df.empty and keyword in google_df.columns:
                google_df = google_df.reset_index().rename(columns={'date': '날짜', keyword: f'Google_{keyword}'})
                all_data.append(google_df[['날짜', f'Google_{keyword}']])

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
                    all_data.append(naver_df)
    
    if not all_data: return pd.DataFrame()
    final_trends_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_data)
    # update_gsheet(gs_client, sheet_name, final_trends_df) # 검색량은 변동성이 크므로 매번 새로 조회
    return final_trends_df

def fetch_kamis_data(gs_client, item_info, start_date, end_date, kamis_keys):
    sheet_name = "kamis"
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
                        all_data.append({'조사일자': date, '도매가격(원)': int(price_str)})
        except Exception: continue
        finally: progress_bar.progress((i + 1) / len(date_range), text=f"KAMIS 데이터 조회 중... {date_str}")
    
    progress_bar.empty()
    if not all_data: st.sidebar.warning("해당 기간에 대한 KAMIS 데이터가 없습니다."); return pd.DataFrame()
    df = pd.DataFrame(all_data)
    # update_gsheet(gs_client, sheet_name, df) # KAMIS도 변동성이 크므로 매번 새로 조회
    return df

# --- Constants ---
COFFEE_TICKERS_YFINANCE = {"미국 커피 C": "KC=F", "런던 로부스타": "RC=F"}
KAMIS_CATEGORIES = {"채소류": "100", "과일류": "200", "축산물": "300", "수산물": "400"}
KAMIS_ITEMS = {"채소류": {"배추": "111", "무": "112", "양파": "114", "마늘": "141"}, "과일류": {"사과": "211", "바나나": "214", "아보카도": "215"}, "축산물": {"소고기": "311", "돼지고기": "312"}, "수산물": {"고등어": "411", "오징어": "413"}}

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 대시보드 (Google Sheets 연동)")

gs_client = get_gspread_client()
if gs_client is None: st.stop()

st.sidebar.header("⚙️ 분석 설정")
st.sidebar.subheader("1. 분석 데이터셋 선택")

raw_trade_df = None
with st.spinner("Google Sheet에서 기존 데이터를 확인 중..."):
    all_tds_data = read_gsheet(gs_client, "TDS")

if not all_tds_data.empty and 'file_name' in all_tds_data.columns:
    existing_files = sorted(all_tds_data['file_name'].unique())
    selection_options = ["새 파일 업로드"] + existing_files
    selected_option = st.sidebar.selectbox("데이터셋 선택 또는 새 파일 업로드", selection_options)

    if selected_option == "새 파일 업로드":
        uploaded_file = st.sidebar.file_uploader("업로드할 신규 파일", type=['csv', 'xlsx'])
        if uploaded_file:
            if uploaded_file.name in existing_files:
                st.sidebar.warning(f"'{uploaded_file.name}'은 이미 존재합니다. 위에서 선택해주세요.")
            else:
                raw_trade_df = process_new_trade_data(gs_client, uploaded_file)
    else:
        st.sidebar.success(f"'{selected_option}' 데이터를 Google Sheet에서 로드했습니다.")
        raw_trade_df = all_tds_data[all_tds_data['file_name'] == selected_option].copy()
        raw_trade_df['Date'] = pd.to_datetime(raw_trade_df['Date'])
        raw_trade_df = raw_trade_df.drop(columns=['file_name'])
else:
    st.sidebar.info("저장된 데이터가 없습니다. 새 파일을 업로드해주세요.")
    uploaded_file = st.sidebar.file_uploader("수출입 데이터 파일 업로드", type=['csv', 'xlsx'])
    if uploaded_file:
        raw_trade_df = process_new_trade_data(gs_client, uploaded_file)

if raw_trade_df is None:
    st.info("👈 사이드바에서 분석할 데이터셋을 선택하거나 새 파일을 업로드해주세요."); st.stop()

# --- Subsequent UI and Logic ---
file_start_date, file_end_date = raw_trade_df['Date'].min(), raw_trade_df['Date'].max()
category_options = sorted(raw_trade_df['Category'].unique())
selected_categories = st.sidebar.multiselect("2. 분석할 품목 카테고리 선택", category_options, default=category_options[0] if category_options else None)
keyword_input = st.sidebar.text_input("3. 검색어 입력 (쉼표로 구분)", ", ".join(selected_categories) if selected_categories else "")
search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
st.sidebar.subheader("4. 분석 기간 설정")
start_date = st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date)
end_date = st.sidebar.date_input('종료일', file_end_date, min_value=start_date, max_value=file_end_date)

# --- External Data Loading Section ---
raw_wholesale_df = pd.DataFrame()
st.sidebar.subheader("🔗 외부 가격 데이터")
is_coffee_selected = any('커피' in str(cat) for cat in selected_categories)

if is_coffee_selected:
    st.sidebar.info("Yahoo Finance에서 선물가격을 가져옵니다.")
    if st.sidebar.button("선물가격 데이터 가져오기"):
        df = fetch_yfinance_data(gs_client, COFFEE_TICKERS_YFINANCE, start_date, end_date)
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
                df = fetch_kamis_data(gs_client, item_info, start_date, end_date, {'key': kamis_api_key, 'id': kamis_api_id})
                st.session_state['wholesale_data'] = df
            else: st.sidebar.error("KAMIS API Key와 ID를 모두 입력해주세요.")

if 'wholesale_data' in st.session_state: raw_wholesale_df = st.session_state['wholesale_data']

# --- Search Data Loading Section ---
raw_search_df = pd.DataFrame()
st.sidebar.subheader("📰 검색량 데이터")
naver_client_id = st.sidebar.text_input("Naver API Client ID", type="password")
naver_client_secret = st.sidebar.text_input("Naver API Client Secret", type="password")
if st.sidebar.button("검색량 데이터 가져오기"):
    if not search_keywords: st.sidebar.warning("검색어를 먼저 입력해주세요.")
    else:
        df = fetch_trends_data(gs_client, search_keywords, start_date, end_date, {'id': naver_client_id, 'secret': naver_client_secret})
        st.session_state['search_data'] = df

if 'search_data' in st.session_state: raw_search_df = st.session_state['search_data']

# --- Main Display Area ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 원본 데이터 확인", "2️⃣ 데이터 표준화", "3️⃣ 최종 통합 데이터"])

with tab1:
    st.subheader("A. 수출입 데이터"); st.dataframe(raw_trade_df.head())
    st.subheader("B. 외부 가격 데이터"); st.dataframe(raw_wholesale_df.head())
    st.subheader("C. 검색량 데이터"); st.dataframe(raw_search_df.head())

with tab2:
    if not selected_categories: st.warning("분석할 카테고리를 선택해주세요.")
    else:
        st.subheader("2-1. 분석 대상 품목 필터링")
        trade_df_in_range = raw_trade_df[(raw_trade_df['Date'] >= pd.to_datetime(start_date)) & (raw_trade_df['Date'] <= pd.to_datetime(end_date))]
        filtered_trade_df = trade_df_in_range[trade_df_in_range['Category'].isin(selected_categories)].copy()
        st.write(f"선택된 카테고리: **{', '.join(selected_categories)}**"); st.dataframe(filtered_trade_df.head())
        
        st.subheader("2-2. 주(Week) 단위 데이터로 집계")
        if not filtered_trade_df.empty:
            filtered_trade_df.set_index('Date', inplace=True)
            trade_weekly = filtered_trade_df.resample('W-Mon').agg({'Value': 'sum', 'Volume': 'sum'})
            trade_weekly['Unit Price'] = trade_weekly['Value'] / trade_weekly['Volume']
            trade_weekly.columns = ['수입액(USD)', '수입량(KG)', '수입단가(USD/KG)']
            
            wholesale_weekly = pd.DataFrame()
            if not raw_wholesale_df.empty:
                date_col = '조사일자' if '조사일자' in raw_wholesale_df.columns else '날짜'
                raw_wholesale_df[date_col] = pd.to_datetime(raw_wholesale_df[date_col])
                wholesale_df_processed = raw_wholesale_df.set_index(date_col)
                price_cols = [col for col in wholesale_df_processed.columns if '가격' in col]
                agg_dict = {col: 'mean' for col in price_cols}
                if agg_dict:
                    wholesale_weekly = wholesale_df_processed.resample('W-Mon').agg(agg_dict)
                    if '도매가격(원)' in wholesale_weekly.columns:
                        wholesale_weekly['도매가격(USD)'] = wholesale_weekly['도매가격(원)'] / 1350
                        wholesale_weekly.drop(columns=['도매가격(원)'], inplace=True)
            
            search_weekly = pd.DataFrame()
            if not raw_search_df.empty:
                raw_search_df['날짜'] = pd.to_datetime(raw_search_df['날짜'])
                search_df_processed = raw_search_df.set_index('날짜')
                numeric_cols = search_df_processed.select_dtypes(include=np.number).columns
                search_weekly = search_df_processed.resample('W-Mon').agg({col: 'mean' for col in numeric_cols})
            
            st.write("▼ 일별(Daily) vs 주별(Weekly) 수입량 비교")
            col1, col2 = st.columns(2)
            with col1: st.line_chart(filtered_trade_df['Volume'])
            with col2: st.line_chart(trade_weekly['수입량(KG)'])
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

