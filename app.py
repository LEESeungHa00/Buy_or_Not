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

# --- Data Fetching and Caching Functions ---
@st.cache_data
def load_trade_data(uploaded_file):
    """
    업로드된 수출입 데이터 파일을 읽고 전처리합니다.
    이 함수는 캐시되어 파일 재로딩으로 인한 딜레이를 방지합니다.
    """
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if 'Date' not in df.columns or 'Category' not in df.columns:
            st.error("업로드된 파일에 'Date'와 'Category' 컬럼이 모두 필요합니다.")
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_yfinance_data(ticker, name, start_date, end_date):
    """Yahoo Finance에서 지정된 티커의 과거 데이터를 가져옵니다."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.warning(f"'{name}({ticker})'에 대한 Yahoo Finance 데이터가 없습니다.")
            return None
        df = data[['Close']].copy()
        df.reset_index(inplace=True)
        df.rename(columns={'Date': '조사일자', 'Close': f'{name} 선물가격(USD)'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Yahoo Finance ('{name}') 데이터를 가져오는 중 오류 발생: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_google_trends(keyword, start_date, end_date):
    """Google Trends에서 지정된 기간의 검색량 데이터를 가져옵니다."""
    pytrends = TrendReq(hl='ko-KR', tz=540)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
    
    try:
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='KR', gprop='')
        df = pytrends.interest_over_time()
        if df.empty or keyword not in df.columns:
            st.warning(f"'{keyword}'에 대한 Google Trends 데이터가 없습니다.")
            return None
        df.reset_index(inplace=True)
        df.rename(columns={'date': '날짜', keyword: f'Google_{keyword}'}, inplace=True)
        return df[['날짜', f'Google_{keyword}']]
    except Exception as e:
        st.error(f"Google Trends ('{keyword}') 데이터를 가져오는 중 오류 발생: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_naver_datalab(client_id, client_secret, keyword, start_date, end_date):
    """Naver DataLab API를 호출하여 검색량 데이터를 가져옵니다."""
    try:
        url = "https://openapi.naver.com/v1/datalab/search"
        body = {
            "startDate": start_date.strftime('%Y-%m-%d'),
            "endDate": end_date.strftime('%Y-%m-%d'),
            "timeUnit": "date",
            "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
        }
        body = json.dumps(body)

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)
        request.add_header("Content-Type", "application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        
        rescode = response.getcode()
        if rescode == 200:
            response_body = response.read()
            result = json.loads(response_body.decode('utf-8'))
            df = pd.DataFrame(result['results'][0]['data'])
            df.rename(columns={'period': '날짜', 'ratio': f'Naver_{keyword}'}, inplace=True)
            df['날짜'] = pd.to_datetime(df['날짜'])
            return df
        else:
            st.error(f"Naver API 오류 발생 ('{keyword}'): Error Code {rescode}")
            return None
    except Exception as e:
        st.error(f"Naver DataLab ('{keyword}') API 호출 중 오류 발생: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_kamis_data(api_key, api_id, item_code, category_code, start_date, end_date):
    """KAMIS에서 일별 품목별 도매가격 데이터를 가져옵니다."""
    all_data = []
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # [수정] 조회 기간이 길 경우 경고 메시지 표시
    if len(date_range) > 180:
        st.sidebar.warning(f"조회 기간이 {len(date_range)}일로 너무 깁니다. 로딩에 매우 오랜 시간이 소요될 수 있습니다.")

    progress_bar = st.sidebar.progress(0, text="KAMIS 데이터 조회 중...")
    
    for i, date in enumerate(date_range):
        date_str = date.strftime('%Y-%m-%d')
        url = (
            "http://www.kamis.or.kr/service/price/xml.do"
            f"?p_product_cls_code=02&p_item_category_code={category_code}"
            f"&p_item_code={item_code}&p_regday={date_str}"
            "&p_convert_kg_yn=Y"
            f"&p_cert_key={api_key}&p_cert_id={api_id}&p_returntype=json"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("data", {}).get("item", [])
            if items:
                price = items[0].get('dpr1', '0').replace(',', '')
                if price and int(price) > 0:
                    all_data.append({'조사일자': date, '도매가격(원)': int(price)})
        except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
            print(f"KAMIS data fetch error for {date_str}: {e}")
        
        progress_bar.progress((i + 1) / len(date_range), text=f"KAMIS 데이터 조회 중... {date_str}")
    
    progress_bar.empty()
    if not all_data:
        st.sidebar.warning("해당 기간에 대한 KAMIS 데이터가 없습니다.")
        return None
        
    df = pd.DataFrame(all_data)
    return df

# --- Constants ---
COFFEE_TICKERS_YFINANCE = {"미국 커피 C": "KC=F", "런던 로부스타": "RC=F"}

# --- [수정] KAMIS 품목 선택지 대폭 확장 ---
KAMIS_CATEGORIES = {
    "채소류": "100", "과일류": "200", "축산물": "300", "수산물": "400"
}
KAMIS_ITEMS = {
    "채소류": {"배추": "111", "무": "112", "양파": "114", "마늘": "141", "오이": "123", "토마토": "126"},
    "과일류": {"사과": "211", "배": "212", "바나나": "214", "아보카도": "215", "오렌지": "223", "레몬": "224"},
    "축산물": {"소고기": "311", "돼지고기": "312", "닭고기": "313", "계란": "314"},
    "수산물": {"고등어": "411", "오징어": "413", "새우": "421", "연어": "423"}
}


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 대시보드")
st.info("각기 다른 소스의 원본 데이터를 확인하고, 이들이 어떻게 하나의 분석용 데이터셋으로 통합되는지 단계별로 살펴봅니다.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ 분석 설정")
uploaded_file = st.sidebar.file_uploader("1. 수출입 데이터 파일 업로드 (CSV or Excel)", type=['csv', 'xlsx'])

# Initialize variables
raw_trade_df, selected_categories, search_keywords = None, [], []
start_date, end_date = None, None

if uploaded_file:
    raw_trade_df = load_trade_data(uploaded_file)
    if raw_trade_df is not None:
        file_start_date, file_end_date = raw_trade_df['Date'].min(), raw_trade_df['Date'].max()
        
        category_options = sorted(raw_trade_df['Category'].unique())
        selected_categories = st.sidebar.multiselect("2. 분석할 품목 카테고리 선택", category_options, default=category_options[0] if category_options else None)
        keyword_input = st.sidebar.text_input("3. 검색어 입력 (쉼표로 구분)", ", ".join(selected_categories) if selected_categories else "")
        search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]

        st.sidebar.subheader("4. 분석 기간 설정")
        start_date = st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date)
        end_date = st.sidebar.date_input('종료일', file_end_date, min_value=start_date, max_value=file_end_date)

    else: st.stop()
else:
    st.info("👈 사이드바에서 분석할 수출입 데이터 파일을 업로드해주세요."); st.stop()

# --- External Data Loading Section ---
raw_wholesale_df = None
st.sidebar.subheader("🔗 외부 가격 데이터")

is_coffee_selected = any('커피' in str(cat) for cat in selected_categories)

if is_coffee_selected:
    st.sidebar.info("커피 관련 품목이 선택되었습니다.\nYahoo Finance에서 선물가격을 가져옵니다.")
    if st.sidebar.button("선물가격 데이터 가져오기 (Yahoo Finance)"):
        all_futures_data = []
        for name, ticker in COFFEE_TICKERS_YFINANCE.items():
             with st.spinner(f"{name}({ticker}) 데이터를 가져오는 중..."):
                data = fetch_yfinance_data(ticker, name, start_date, end_date)
                if data is not None: all_futures_data.append(data)
        if all_futures_data:
            st.session_state['futures_data'] = reduce(lambda left, right: pd.merge(left, right, on='조사일자', how='outer'), all_futures_data)
            st.sidebar.success("선물가격 데이터 로드 성공!")
        else: st.sidebar.error("선물가격 데이터를 가져오지 못했습니다.")
    if 'futures_data' in st.session_state:
        raw_wholesale_df = st.session_state['futures_data'].sort_values('조사일자')
else:
    st.sidebar.info("KAMIS에서 농산물 도매가격 데이터를 가져옵니다.")
    kamis_api_key = st.sidebar.text_input("KAMIS API Key", value="9b2f72e0-1909-4c08-8f4f-9b6f55b44c88", type="password")
    kamis_api_id = st.sidebar.text_input("KAMIS API ID", type="password")
    
    selected_kamis_category_name = st.sidebar.selectbox("품목 분류 선택", list(KAMIS_CATEGORIES.keys()))
    if selected_kamis_category_name:
        kamis_category_code = KAMIS_CATEGORIES[selected_kamis_category_name]
        item_options = KAMIS_ITEMS[selected_kamis_category_name]
        selected_item_name = st.sidebar.selectbox("세부 품목 선택", list(item_options.keys()))
        kamis_item_code = item_options[selected_item_name]

    if st.sidebar.button("KAMIS 데이터 가져오기"):
        if kamis_api_key and kamis_api_id:
            kamis_df = fetch_kamis_data(kamis_api_key, kamis_api_id, kamis_item_code, kamis_category_code, start_date, end_date)
            st.session_state['kamis_data'] = kamis_df
            if kamis_df is not None: st.sidebar.success("KAMIS 데이터 로드 성공!")
        else: st.sidebar.error("KAMIS API Key와 ID를 모두 입력해주세요.")
    if 'kamis_data' in st.session_state: raw_wholesale_df = st.session_state['kamis_data']

# --- Search/News Data Loading Section ---
st.sidebar.subheader("📰 검색량/뉴스 데이터")
naver_client_id = st.sidebar.text_input("Naver API Client ID", type="password")
naver_client_secret = st.sidebar.text_input("Naver API Client Secret", type="password")

if st.sidebar.button("Google/Naver 데이터 가져오기"):
    if not search_keywords: st.sidebar.warning("검색어를 먼저 입력해주세요.")
    else:
        all_trends_data = []
        for keyword in search_keywords:
            with st.spinner(f"'{keyword}' Google Trends 데이터 가져오는 중..."):
                google_data = fetch_google_trends(keyword, start_date, end_date)
                if google_data is not None: all_trends_data.append(google_data)
            if naver_client_id and naver_client_secret:
                with st.spinner(f"'{keyword}' Naver DataLab 데이터 가져오는 중..."):
                    naver_data = fetch_naver_datalab(naver_client_id, naver_client_secret, keyword, start_date, end_date)
                    if naver_data is not None: all_trends_data.append(naver_data)
        if all_trends_data:
            st.session_state['search_data'] = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_trends_data)
            st.sidebar.success("검색량 데이터 로드 완료!")
        else: st.sidebar.error("검색량 데이터를 가져오지 못했습니다.")

raw_search_df = st.session_state.get('search_data', pd.DataFrame({'날짜': pd.to_datetime([])}))
if not raw_search_df.empty: raw_search_df.sort_values('날짜', inplace=True)

# --- Main Display Area ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 원본 데이터 확인", "2️⃣ 데이터 표준화 (Preprocessing)", "3️⃣ 최종 통합 데이터"])

with tab1:
    st.header("1. 각기 다른 데이터 소스의 원본 형태")
    st.subheader("A. 수출입 데이터 (사용자 제공)"); st.dataframe(raw_trade_df.head())
    st.subheader("B. 외부 가격 데이터 (선물/도매)")
    if raw_wholesale_df is not None: st.dataframe(raw_wholesale_df.head())
    else: st.warning("사이드바에서 외부 가격 데이터를 가져오거나 업로드해주세요.")
    st.subheader("C. 검색량 데이터 (Google/Naver)")
    if not raw_search_df.empty: st.dataframe(raw_search_df.head())
    else: st.warning("사이드바에서 검색량 데이터를 가져오세요.")

with tab2:
    st.header("2. 데이터 표준화: 같은 기준으로 데이터 맞춰주기")
    if not selected_categories: st.warning("분석할 카테고리를 사이드바에서 하나 이상 선택해주세요.")
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
            if raw_wholesale_df is not None:
                wholesale_df_processed = raw_wholesale_df.set_index('조사일자')
                price_cols = [col for col in wholesale_df_processed.columns if '가격' in col]
                agg_dict = {col: 'mean' for col in price_cols}
                if agg_dict:
                    wholesale_weekly = wholesale_df_processed.resample('W-Mon').agg(agg_dict)
                    if '도매가격(원)' in wholesale_weekly.columns:
                        wholesale_weekly['도매가격(USD)'] = wholesale_weekly['도매가격(원)'] / 1350
                        wholesale_weekly.drop(columns=['도매가격(원)'], inplace=True)
            
            search_weekly = pd.DataFrame()
            if not raw_search_df.empty:
                search_df_processed = raw_search_df.set_index('날짜')
                numeric_cols = search_df_processed.select_dtypes(include=np.number).columns
                search_weekly = search_df_processed.resample('W-Mon').agg({col: 'mean' for col in numeric_cols})
            
            st.write("▼ 일별(Daily) 데이터가 주별(Weekly) 데이터로 집계된 결과")
            col1, col2 = st.columns(2)
            with col1: st.write("**Before (일별 수입량)**"); st.line_chart(filtered_trade_df['Volume'])
            with col2: st.write("**After (주별 수입량)**"); st.line_chart(trade_weekly['수입량(KG)'])
        else: st.warning("선택된 카테고리에 해당하는 데이터가 없습니다.")

with tab3:
    st.header("3. 최종 통합 데이터셋")
    if 'trade_weekly' in locals() and not trade_weekly.empty:
        dfs_to_concat = [trade_weekly]
        if 'wholesale_weekly' in locals() and not wholesale_weekly.empty: dfs_to_concat.append(wholesale_weekly)
        if 'search_weekly' in locals() and not search_weekly.empty: dfs_to_concat.append(search_weekly)
        
        final_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs_to_concat)
        final_df = final_df.dropna(thresh=2).fillna(method='ffill')
        
        st.dataframe(final_df)
        st.subheader("최종 데이터 시각화")
        fig = px.line(final_df, labels={'value': '값', 'index': '날짜', 'variable': '데이터 종류'}, title="최종 통합 데이터 시계열 추이")
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("통합할 데이터가 없습니다. 2단계 표준화 과정을 먼저 확인해주세요.")

