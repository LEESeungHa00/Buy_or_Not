import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from functools import reduce
from pytrends.request import TrendReq
import json
import urllib.request

# --- Web Scraping Function ---
@st.cache_data(ttl=3600) # 데이터를 1시간 동안 캐싱
def fetch_investing_data(index_name, url):
    """investing.com에서 지정된 상품의 과거 데이터를 스크래핑합니다."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'data-test': 'historical-data-table'})
        if not table:
            st.error(f"Investing.com 페이지({index_name})에서 데이터 테이블을 찾지 못했습니다.")
            return None

        dates, prices = [], []
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) > 1:
                date_str = cells[0].find('time')['datetime']
                price_str = cells[1].text.strip().replace(',', '')
                dates.append(pd.to_datetime(date_str))
                prices.append(float(price_str))

        column_name = f'{index_name} 선물가격(USD)'
        df = pd.DataFrame({'조사일자': dates, column_name: prices})
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"웹사이트({index_name})에 접속하는 중 오류 발생: {e}")
        return None
    except Exception as e:
        st.error(f"데이터({index_name})를 파싱하는 중 오류 발생: {e}")
        return None

# --- Real-time Data Fetching Functions ---
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
        df.rename(columns={'date': '날짜', keyword: 'Google 검색량'}, inplace=True)
        return df[['날짜', 'Google 검색량']]
    except Exception as e:
        st.error(f"Google Trends 데이터를 가져오는 중 오류 발생: {e}")
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
            df.rename(columns={'period': '날짜', 'ratio': 'Naver 검색량'}, inplace=True)
            df['날짜'] = pd.to_datetime(df['날짜'])
            return df
        else:
            st.error(f"Naver API 오류 발생: Error Code {rescode}")
            return None
    except Exception as e:
        st.error(f"Naver DataLab API 호출 중 오류 발생: {e}")
        return None

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 대시보드")
st.info("각기 다른 소스의 원본 데이터를 확인하고, 이들이 어떻게 하나의 분석용 데이터셋으로 통합되는지 단계별로 살펴봅니다.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ 분석 설정")
uploaded_file = st.sidebar.file_uploader("수출입 데이터 파일 업로드 (CSV or Excel)", type=['csv', 'xlsx'])

st.sidebar.subheader("분석 대표 품목")
selected_product_category = st.sidebar.selectbox("분석할 대표 품목 선택", ['인스턴트 커피', '원두 커피', '캡슐 커피', '아보카도'])

# --- Keyword Mapping & Constants ---
KEYWORD_MAPPING = {
    '맥심 모카골드': '인스턴트 커피', '스타벅스 파이크플레이스': '원두 커피', '네스카페 돌체구스토': '캡슐 커피',
    '커피': '인스턴트 커피', '인스턴트 커피': '인스턴트 커피', '아보카도': '아보카도'
}
COFFEE_INDICES = {
    "런던 커피": "https://kr.investing.com/commodities/london-coffee-historical-data",
    "미국 커피 C": "https://kr.investing.com/commodities/us-coffee-c-historical-data"
}

# --- Data Loading Logic ---
raw_trade_df = None
if uploaded_file:
    try:
        raw_trade_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        if 'Date' in raw_trade_df.columns:
            raw_trade_df['Date'] = pd.to_datetime(raw_trade_df['Date'])
            start_date = raw_trade_df['Date'].min()
            end_date = raw_trade_df['Date'].max()
        else:
            st.error("업로드된 파일에 'Date' 컬럼이 없습니다.")
            st.stop()
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    st.info("👈 사이드바에서 분석할 수출입 데이터 파일을 업로드해주세요.")
    st.stop()

# External Price Data Loading
raw_wholesale_df = None
st.sidebar.subheader("🔗 외부 가격 데이터")
if '커피' in selected_product_category:
    st.sidebar.info("커피 품목이 선택되었습니다.\nInvesting.com에서 선물가격을 가져옵니다.")
    for name, url in COFFEE_INDICES.items():
        if st.sidebar.button(f"{name} 선물가격 가져오기"):
            with st.spinner(f"Investing.com에서 {name} 데이터를 스크래핑 중..."):
                data = fetch_investing_data(name, url)
                if data is not None:
                    st.session_state[f'{name}_data'] = data
                    st.sidebar.success(f"{name} 데이터 로드 성공!")
    loaded_futures_dfs = [st.session_state[f'{name}_data'] for name in COFFEE_INDICES if f'{name}_data' in st.session_state]
    if loaded_futures_dfs:
        raw_wholesale_df = reduce(lambda left, right: pd.merge(left, right, on='조사일자', how='outer'), loaded_futures_dfs)
        raw_wholesale_df.sort_values('조사일자', inplace=True)
else:
    wholesale_data_file = st.sidebar.file_uploader("도매가격 데이터 업로드 (KAMIS 등)", type=['csv', 'xlsx'])
    # ... (file upload logic for non-coffee items) ...

# Search/News Data Loading
st.sidebar.subheader("📰 검색량/뉴스 데이터")
naver_client_id = st.sidebar.text_input("Naver API Client ID", type="password")
naver_client_secret = st.sidebar.text_input("Naver API Client Secret", type="password")

if st.sidebar.button("Google/Naver 데이터 가져오기"):
    with st.spinner("Google Trends 데이터 가져오는 중..."):
        google_data = fetch_google_trends(selected_product_category, start_date, end_date)
        if google_data is not None: st.session_state['google_trends_data'] = google_data
    
    if naver_client_id and naver_client_secret:
         with st.spinner("Naver DataLab 데이터 가져오는 중..."):
            naver_data = fetch_naver_datalab(naver_client_id, naver_client_secret, selected_product_category, start_date, end_date)
            if naver_data is not None: st.session_state['naver_search_data'] = naver_data
    else:
        st.sidebar.warning("Naver API 키를 입력하면 Naver 검색량도 함께 가져옵니다.")

# Merge Search Data
loaded_search_dfs = []
if 'google_trends_data' in st.session_state: loaded_search_dfs.append(st.session_state['google_trends_data'])
if 'naver_search_data' in st.session_state: loaded_search_dfs.append(st.session_state['naver_search_data'])

raw_search_df = pd.DataFrame({'날짜': pd.to_datetime([])})
if loaded_search_dfs:
    raw_search_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), loaded_search_dfs)
    raw_search_df.sort_values('날짜', inplace=True)

# --- Display Tabs ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 원본 데이터 확인", "2️⃣ 데이터 표준화 (Preprocessing)", "3️⃣ 최종 통합 데이터"])

with tab1:
    st.header("1. 각기 다른 데이터 소스의 원본 형태")
    st.subheader("A. 수출입 데이터 (사용자 제공)")
    st.dataframe(raw_trade_df.head())
    st.subheader("B. 외부 가격 데이터 (선물/도매)")
    if raw_wholesale_df is not None: st.dataframe(raw_wholesale_df.head())
    else: st.warning("사이드바에서 외부 가격 데이터를 가져오거나 업로드해주세요.")
    st.subheader("C. 검색량 데이터 (Google/Naver)")
    if not raw_search_df.empty: st.dataframe(raw_search_df.head())
    else: st.warning("사이드바에서 검색량 데이터를 가져오세요.")

with tab2:
    st.header("2. 데이터 표준화: 같은 기준으로 데이터 맞춰주기")
    st.subheader("2-1. 품목 이름 통합 (Keyword Mapping)")
    filtered_trade_df = raw_trade_df.copy()
    if 'Reported Product Name' in filtered_trade_df.columns:
        filtered_trade_df['대표 품목'] = filtered_trade_df['Reported Product Name'].map(KEYWORD_MAPPING)
        filtered_trade_df = filtered_trade_df[filtered_trade_df['대표 품목'] == selected_product_category]
        if not filtered_trade_df.empty:
            st.dataframe(filtered_trade_df[['Date', 'Reported Product Name', '대표 품목', 'Value', 'Volume']].head())
        else:
            st.warning(f"수출입 데이터에서 '{selected_product_category}'에 해당하는 품목을 찾을 수 없습니다.")
    
    st.subheader("2-2. 주(Week) 단위 데이터로 집계")
    if not filtered_trade_df.empty:
        filtered_trade_df = filtered_trade_df.set_index('Date')
        trade_weekly = filtered_trade_df.resample('W-Mon').agg({'Value': 'sum', 'Volume': 'sum'})
        trade_weekly['Unit Price'] = trade_weekly['Value'] / trade_weekly['Volume']
        trade_weekly.columns = ['수입액(USD)', '수입량(KG)', '수입단가(USD/KG)']

        wholesale_weekly = pd.DataFrame()
        if raw_wholesale_df is not None:
            # ... (Wholesale weekly aggregation logic) ...
            wholesale_df_processed = raw_wholesale_df.set_index('조사일자')
            price_cols = [col for col in wholesale_df_processed.columns if '가격' in col]
            agg_dict = {col: 'mean' for col in price_cols}
            if agg_dict:
                wholesale_weekly = wholesale_df_processed.resample('W-Mon').agg(agg_dict)
                if '도매가격(원)' in wholesale_weekly.columns:
                    wholesale_weekly['도매가격(USD)'] = wholesale_weekly['도매가격(원)'] / 1350
                    wholesale_weekly = wholesale_weekly.drop(columns=['도매가격(원)'])
        
        search_weekly = pd.DataFrame()
        if not raw_search_df.empty:
            search_df_processed = raw_search_df.set_index('날짜')
            numeric_cols = search_df_processed.select_dtypes(include=np.number).columns
            search_weekly = search_df_processed.resample('W-Mon').agg({col: 'mean' for col in numeric_cols})
        
        col1, col2 = st.columns(2)
        with col1: st.write("**Before (일별 수입량)**"); st.line_chart(filtered_trade_df['Volume'])
        with col2: st.write("**After (주별 수입량)**"); st.line_chart(trade_weekly['수입량(KG)'])

with tab3:
    st.header("3. 최종 통합 데이터셋")
    if 'trade_weekly' in locals() and not trade_weekly.empty:
        dfs_to_concat = [trade_weekly]
        if 'wholesale_weekly' in locals() and not wholesale_weekly.empty: dfs_to_concat.append(wholesale_weekly)
        if 'search_weekly' in locals() and not search_weekly.empty: dfs_to_concat.append(search_weekly)
        
        final_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs_to_concat)
        final_df = final_df.dropna(thresh=2)
        
        st.dataframe(final_df)
        st.subheader("최종 데이터 시각화")
        fig = px.line(final_df, labels={'value': '값', 'index': '날짜', 'variable': '데이터 종류'}, title="최종 통합 데이터 시계열 추이")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("통합할 데이터가 없습니다. 2단계 표준화 과정을 먼저 확인해주세요.")

