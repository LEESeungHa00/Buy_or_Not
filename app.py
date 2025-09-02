import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from datetime import datetime

# --- KAMIS API Codes ---
# 사용자가 선택할 수 있도록 품목 코드 목록을 미리 정의합니다.
KAMIS_CATEGORIES = {
    '가공식품': '300',
    '채소류': '200',
    '과일류': '400',
    '식량작물': '100',
}
KAMIS_ITEMS = {
    '300': {'커피': '314', '라면': '315', '설탕': '324', '식용유': '316'},
    '200': {'배추': '211', '양파': '223', '마늘': '225', '고추': '243'},
    '400': {'사과': '411', '배': '412', '바나나': '418'},
    '100': {'쌀': '111', '찹쌀': '112', '콩': '131'},
}

# --- API Data Fetching Function ---

def fetch_kamis_data(api_key, cert_id, start_date, end_date, category_code, item_code):
    """KAMIS API를 호출하여 지정된 기간의 도매가 데이터를 가져옵니다."""
    if not api_key or not cert_id:
        st.sidebar.warning("KAMIS API 키와 ID를 입력해주세요.")
        return None

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    url = "http://www.kamis.or.kr/service/price/xml.do"
    params = {
        'p_product_cls_code': '02',
        'p_startday': start_str,
        'p_endday': end_str,
        'p_itemcategorycode': category_code,
        'p_itemcode': item_code,
        'p_kindcode': '00',
        'p_productrankcode': '04',
        'p_countrycode': '1101',
        'p_cert_key': api_key,
        'p_cert_id': cert_id,
        'p_returntype': 'json'
    }

    try:
        with st.spinner(f"KAMIS API에서 '{list(KAMIS_ITEMS[category_code].keys())[list(KAMIS_ITEMS[category_code].values()).index(item_code)]}' 데이터를 가져오는 중..."):
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'data' in data and 'item' in data['data']:
                items = data['data']['item']
                if not items: # 데이터가 없는 경우
                    st.sidebar.warning("선택하신 기간/품목에 대한 KAMIS 데이터가 없습니다.")
                    return pd.DataFrame() # 빈 데이터프레임 반환
                
                df = pd.DataFrame(items)
                df = df[['regday', 'price']]
                df.columns = ['조사일자', '도매가격(원)']
                
                df['조사일자'] = pd.to_datetime(df['조사일자'], format='%Y/%m/%d')
                df['도매가격(원)'] = pd.to_numeric(df['price'].str.replace(',', ''), errors='coerce')
                
                st.sidebar.success("KAMIS 데이터 로드 성공!")
                return df
            else:
                error_msg = data.get('error_message', '데이터 없음')
                st.sidebar.error(f"KAMIS API 오류: {error_msg}")
                return None
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"API 호출 중 네트워크 오류: {e}")
        return None
    except Exception as e:
        st.sidebar.error(f"데이터 처리 중 오류: {e}")
        return None

# --- Mock Data Generation Function ---
def create_mock_search_data():
    """Naver/Google 검색량 데이터의 가상 버전 생성"""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", end="2024-12-31", freq='D'))
    n = len(dates)
    data = {
        '날짜': dates,
        '키워드': '인스턴트 커피',
        '검색량': [70 + np.sin(i/28)*25 + np.random.randint(-5, 5) for i in range(n)]
    }
    return pd.DataFrame(data)

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 대시보드")
st.info("각기 다른 소스의 원본 데이터를 확인하고, 이들이 어떻게 하나의 분석용 데이터셋으로 통합되는지 단계별로 살펴봅니다.")

# --- Sidebar for file upload and controls ---
st.sidebar.header("⚙️ 분석 설정")
uploaded_file = st.sidebar.file_uploader("수출입 데이터 파일 업로드 (CSV or Excel)", type=['csv', 'xlsx'])

st.sidebar.subheader("🔗 외부 데이터 연동 (API)")
kamis_api_key = st.sidebar.text_input("KAMIS API 인증키", type="password", help="kamis.or.kr에서 발급받은 인증키를 입력하세요.")
kamis_cert_id = st.sidebar.text_input("KAMIS API 인증 ID", type="password", help="API 신청 시 등록한 ID를 입력하세요.")

st.sidebar.subheader("KAMIS 품목 선택")
selected_category_name = st.sidebar.selectbox("품목 분류", list(KAMIS_CATEGORIES.keys()))
selected_category_code = KAMIS_CATEGORIES[selected_category_name]

available_items = KAMIS_ITEMS[selected_category_code]
selected_item_name = st.sidebar.selectbox("세부 품목", list(available_items.keys()))
selected_item_code = available_items[selected_item_name]

st.sidebar.subheader("분석 대표 품목")
selected_product_category = st.sidebar.selectbox("분석할 대표 품목 선택", ['인스턴트 커피', '원두 커피', '캡슐 커피'])


# --- Keyword Mapping Rule ---
KEYWORD_MAPPING = {
    '맥심 모카골드': '인스턴트 커피',
    '스타벅스 파이크플레이스': '원두 커피',
    '네스카페 돌체구스토': '캡슐 커피',
    '커피': '인스턴트 커피', # KAMIS 데이터용
    '인스턴트 커피': '인스턴트 커피' # 검색 데이터용
}

# --- Main App Logic ---

# 1. 데이터 로드
raw_trade_df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            raw_trade_df = pd.read_csv(uploaded_file)
        else:
            raw_trade_df = pd.read_excel(uploaded_file)
        
        if 'Date' in raw_trade_df.columns:
            raw_trade_df['Date'] = pd.to_datetime(raw_trade_df['Date'])
        else:
            st.error("업로드된 파일에 'Date' 컬럼이 없습니다. 날짜 정보가 담긴 컬럼의 이름을 'Date'로 변경해주세요.")
            st.stop()
            
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    st.info("👈 사이드바에서 분석할 수출입 데이터 파일을 업로드해주세요.")
    st.stop()

# 외부 데이터 로드
min_date = raw_trade_df['Date'].min().date()
max_date = raw_trade_df['Date'].max().date()
raw_kamis_df = fetch_kamis_data(kamis_api_key, kamis_cert_id, min_date, max_date, selected_category_code, selected_item_code)
raw_search_df = create_mock_search_data()


# Tabs
tab1, tab2, tab3 = st.tabs(["1️⃣ 원본 데이터 확인", "2️⃣ 데이터 표준화 (Preprocessing)", "3️⃣ 최종 통합 데이터"])

with tab1:
    st.header("1. 각기 다른 데이터 소스의 원본 형태")
    st.write("API, 크롤링, 엑셀 파일 등에서 가져온 데이터는 아래처럼 서로 다른 형식과 구조를 가집니다.")
    
    st.subheader("A. 수출입 데이터 (사용자 제공)")
    st.dataframe(raw_trade_df.head())

    st.subheader("B. KAMIS 도소매가 데이터")
    if raw_kamis_df is not None and not raw_kamis_df.empty:
        st.dataframe(raw_kamis_df.head())
    else:
        st.warning("KAMIS API 키와 ID를 사이드바에 입력하면 실제 데이터를 가져옵니다. 현재는 데이터가 없습니다.")

    st.subheader("C. 검색량 데이터 (Naver/Google)")
    st.dataframe(raw_search_df.head())

with tab2:
    st.header("2. 데이터 표준화: 같은 기준으로 데이터 맞춰주기")
    st.write("분석을 위해 모든 데이터를 '주(Week)' 단위로 맞추고, 품목 이름을 통일하는 등의 전처리 과정을 거칩니다.")

    st.subheader("2-1. 품목 이름 통합 (Keyword Mapping)")
    st.write(f"**'{selected_product_category}'** 와 관련된 여러 이름들을 하나의 대표 이름으로 통합합니다.")
    
    filtered_trade_df = raw_trade_df.copy()
    if 'Reported Product Name' in filtered_trade_df.columns:
        filtered_trade_df['대표 품목'] = filtered_trade_df['Reported Product Name'].map(KEYWORD_MAPPING)
        filtered_trade_df = filtered_trade_df[filtered_trade_df['대표 품목'] == selected_product_category]
        st.write("▼ 수출입 데이터에서 'Reported Product Name'이 '대표 품목'으로 통합된 결과")
        st.dataframe(filtered_trade_df[['Date', 'Reported Product Name', '대표 품목', 'Value', 'Volume']].head())
    else:
        st.warning("'Reported Product Name' 컬럼이 없어 품목 이름 통합을 건너뜁니다.")

    
    st.subheader("2-2. 주(Week) 단위 데이터로 집계")
    st.write("모든 데이터를 '매주 월요일' 기준으로 합산하거나 평균을 내어 주별 데이터로 변환합니다.")

    filtered_trade_df = filtered_trade_df.set_index('Date')
    trade_weekly = filtered_trade_df.resample('W-Mon').agg({'Value': 'sum', 'Volume': 'sum'})
    trade_weekly['Unit Price'] = trade_weekly['Value'] / trade_weekly['Volume']
    trade_weekly.columns = ['수입액(USD)', '수입량(KG)', '수입단가(USD/KG)']

    kamis_weekly = pd.DataFrame()
    if raw_kamis_df is not None and not raw_kamis_df.empty:
        kamis_df_processed = raw_kamis_df.set_index('조사일자')
        kamis_weekly = kamis_df_processed.resample('W-Mon').agg({'도매가격(원)': 'mean'})
        kamis_weekly['도매가격(USD)'] = kamis_weekly['도매가격(원)'] / 1350 # 환율 가정

    search_df_processed = raw_search_df.set_index('날짜')
    search_weekly = search_df_processed.resample('W-Mon').agg({'검색량': 'mean'})
    
    st.write("▼ 일별(Daily) 데이터가 주별(Weekly) 데이터로 집계된 결과")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Before (일별 수입량)**")
        st.line_chart(filtered_trade_df['Volume'])
    with col2:
        st.write("**After (주별 수입량)**")
        st.line_chart(trade_weekly['수입량(KG)'])


with tab3:
    st.header("3. 최종 통합 데이터셋")
    st.write("모든 표준화 과정을 거친 데이터들을 '날짜'를 기준으로 합쳐, 분석에 사용할 최종 데이터셋을 완성합니다.")

    dfs_to_concat = [trade_weekly, search_weekly]
    if not kamis_weekly.empty:
        dfs_to_concat.append(kamis_weekly)

    final_df = pd.concat(dfs_to_concat, axis=1)
    
    # NaN이 많은 초기 데이터 제거
    final_df = final_df.dropna(thresh=2)
    
    final_columns = ['수입량(KG)', '수입단가(USD/KG)', '검색량']
    if '도매가격(USD)' in final_df.columns:
        final_columns.insert(2, '도매가격(USD)')
    
    # 존재하는 컬럼만 선택
    final_df = final_df[[col for col in final_columns if col in final_df.columns]]

    st.write("이 통합된 데이터가 `analysis_app.py`에서 상관관계를 분석하는 데 사용됩니다.")
    st.dataframe(final_df)

    st.subheader("최종 데이터 시각화")
    st.write("통합된 각 데이터 항목의 시계열 추세를 한눈에 확인합니다.")
    
    fig = px.line(final_df, x=final_df.index, y=final_df.columns,
                  labels={'value': '값', 'date': '날짜', 'variable': '데이터 종류'},
                  title="최종 통합 데이터 시계열 추이")
    st.plotly_chart(fig, use_container_width=True)

