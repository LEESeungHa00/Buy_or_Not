import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import io
from datetime import datetime, timedelta

# 구글 시트 API 연동을 위한 라이브러리
import gspread
from google.oauth2 import service_account

# 설정: 페이지 제목 및 레이아웃
st.set_page_config(
    page_title="Data-Driven_Direction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'df_imports' not in st.session_state:
    st.session_state.df_imports = pd.DataFrame()
if 'df_naver' not in st.session_state:
    st.session_state.df_naver = pd.DataFrame()
if 'df_tds' not in st.session_state:
    st.session_state.df_tds = pd.DataFrame()
if 'df_combined' not in st.session_state:
    st.session_state.df_combined = pd.DataFrame()
if 'selected_hscodes' not in st.session_state:
    st.session_state.selected_hscodes = []

st.title("🧭 Compass - Data-Driven Direction")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------
# 구글 시트 연동 함수
# -----------------
@st.cache_resource(ttl=3600)
def get_google_sheet_client():
    """
    Streamlit secrets를 사용하여 구글 시트 클라이언트를 인증하고 가져오는 함수.
    실제 Streamlit Cloud 환경에서 secrets에 서비스 계정 JSON이 설정되어 있어야 함.
    """
    try:
        credentials = service_account.Credentials.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(credentials)
        return gc
    except Exception as e:
        st.error(f"구글 시트 인증 오류: {e}")
        return None

def read_google_sheet(sheet_name):
    """
    지정된 구글 시트의 워크시트에서 데이터를 읽어 DataFrame으로 반환합니다.
    """
    gc = get_google_sheet_client()
    if gc:
        try:
            sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/12YdcKX3nvaNfFWYkJApRnoKAQnjCeR09AGRJ6rBiOuM/edit?gid=0#gid=0")
            worksheet = sh.worksheet(sheet_name)
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            
            # 네이버 데이터랩 시트일 경우, '커피' 컬럼 이름을 '검색량'으로 변경
            if sheet_name == '네이버 데이터랩' and '커피' in df.columns:
                df.rename(columns={'커피': '검색량'}, inplace=True)
            
            # TDS 시트일 경우, 'Detailed HS-CODE' 컬럼명을 'HS코드'로 변경
            if sheet_name == 'TDS' and 'Detailed HS-CODE' in df.columns:
                df.rename(columns={'Detailed HS-CODE': 'HS코드'}, inplace=True)
            return df
        except Exception as e:
            st.error(f"'{sheet_name}' 워크시트 읽기 오류: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# -----------------
# 파일 업로드 및 데이터 처리
# -----------------
st.sidebar.header("데이터 업로드 및 가져오기")
uploaded_imports = st.sidebar.file_uploader("1. 관세청 데이터 (.csv)", type="csv", key="imports")
uploaded_naver = st.sidebar.file_uploader("2. 네이버 데이터랩 (.csv)", type="csv", key="naver")
uploaded_tds = st.sidebar.file_uploader("3. 트릿지 데이터 (.csv)", type="csv", key="tds")

def load_data():
    if uploaded_imports:
        try:
            df = pd.read_csv(uploaded_imports)
            if '기간' not in df.columns:
                if '년' in df.columns and '월' in df.columns:
                    df['기간'] = df['년'].astype(str) + '.' + df['월'].astype(str).str.zfill(2)
            st.session_state.df_imports = pd.concat([st.session_state.df_imports, df], ignore_index=True)
            st.sidebar.success("관세청 데이터 업로드 완료!")
        except Exception as e:
            st.sidebar.error(f"관세청 CSV 파일 형식이 올바르지 않습니다: {e}")

    if uploaded_naver:
        try:
            df = pd.read_csv(uploaded_naver, skiprows=6)
            df.columns = ['날짜', '검색량']
            st.session_state.df_naver = pd.concat([st.session_state.df_naver, df], ignore_index=True)
            st.sidebar.success("네이버 데이터랩 업로드 완료!")
        except Exception as e:
            st.sidebar.error(f"네이버 데이터랩 CSV 파일 형식이 올바르지 않습니다: {e}")

    if uploaded_tds:
        try:
            df = pd.read_csv(uploaded_tds)
            st.session_state.df_tds = pd.concat([st.session_state.df_tds, df], ignore_index=True)
            st.sidebar.success("TDS 업로드 완료!")
        except Exception as e:
            st.sidebar.error(f"TDS CSV 파일 형식이 올바르지 않습니다: {e}")

if st.sidebar.button("데이터 업로드 및 가져오기"):
    load_data()
    st.session_state.df_imports = read_google_sheet("관세청")
    st.session_state.df_naver = read_google_sheet("네이버 데이터랩")
    st.session_state.df_tds = read_google_sheet("TDS")
    if not st.session_state.df_imports.empty: st.sidebar.success("관세청 데이터 불러오기 완료!")
    if not st.session_state.df_naver.empty: st.sidebar.success("네이버 데이터랩 데이터 불러오기 완료!")
    if not st.session_state.df_tds.empty: st.sidebar.success("TDS 데이터 불러오기 완료!")

if st.session_state.df_imports.empty or st.session_state.df_tds.empty or st.session_state.df_naver.empty:
    st.warning("분석을 시작하려면 먼저 사이드바에서 **데이터 업로드 및 가져오기** 버튼을 눌러주세요.")
else:
    # HS코드 선택 기능
    all_hscodes = pd.concat([
        st.session_state.df_imports[['HS코드', '품목명']],
        st.session_state.df_tds[['HS코드', 'Product Description']]
    ]).drop_duplicates().sort_values(by='HS코드').reset_index(drop=True)
    
    if 'Product Description' in all_hscodes.columns:
        all_hscodes.rename(columns={'Product Description': '품목명'}, inplace=True)
    
    all_hscodes = all_hscodes.drop_duplicates(subset=['HS코드'])
    
    hscode_options = [f"{row['HS코드']} - {row['품목명']}" for index, row in all_hscodes.iterrows()]
    st.session_state.selected_hscodes = st.sidebar.multiselect(
        "분석할 HS코드를 선택하세요",
        options=hscode_options,
        default=hscode_options[:2] if len(hscode_options) > 1 else hscode_options
    )

    selected_codes = [s.split(' - ')[0] for s in st.session_state.selected_hscodes]

    if not selected_codes:
        st.warning("분석을 위해 최소 하나 이상의 HS코드를 선택해야 합니다.")
    else:
        # 데이터 결합
        with st.spinner('데이터를 통합하는 중입니다...'):
            try:
                # 관세청 데이터 필터링 및 전처리
                df_imports_filtered = st.session_state.df_imports[
                    st.session_state.df_imports['HS코드'].astype(str).isin(selected_codes)
                ].copy()
                df_imports_filtered['기간'] = pd.to_datetime(df_imports_filtered['기간'], format='%Y.%m', errors='coerce')
                df_imports_filtered.dropna(subset=['기간'], inplace=True)
                df_imports_monthly = df_imports_filtered.groupby(
                    pd.Grouper(key='기간', freq='M')
                ).agg({
                    '수입 중량': 'sum',
                    '수입 금액': 'sum'
                }).reset_index()

                # 네이버 데이터랩 데이터 전처리
                df_naver_monthly = st.session_state.df_naver.copy()
                df_naver_monthly['날짜'] = pd.to_datetime(df_naver_monthly['날짜'], errors='coerce')
                df_naver_monthly.dropna(subset=['날짜'], inplace=True)
                df_naver_monthly = df_naver_monthly.groupby(
                    pd.Grouper(key='날짜', freq='M')
                ).agg(
                    {'검색량': 'mean'}
                ).reset_index()
                
                # 최종 데이터 병합
                st.session_state.df_combined = pd.merge(
                    df_imports_monthly, 
                    df_naver_monthly, 
                    left_on=df_imports_monthly['기간'].dt.strftime('%Y-%m'), 
                    right_on=df_naver_monthly['날짜'].dt.strftime('%Y-%m'),
                    how='inner'
                )
                st.session_state.df_combined.rename(columns={'key_0': '기간'}, inplace=True)
                st.session_state.df_combined.drop(['날짜'], axis=1, inplace=True)
                st.success("데이터 통합 완료!")
            except Exception as e:
                st.error(f"데이터 통합 중 오류가 발생했습니다. 선택한 HS코드에 해당하는 데이터가 없거나, 업로드한 파일 형식을 확인해주세요: {e}")

        # -----------------
        # 탭 구성
        # -----------------
        tab1, tab2, tab3, tab4 = st.tabs(["📊 대시보드", "🔮 예측 모델", "📈 상관관계 분석", "🗃️ 원본 데이터"])

        with tab1:
            st.header("커피 원두 시장 동향 분석")
            if not st.session_state.df_combined.empty:
                # KPI 지표
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_volume = st.session_state.df_combined['수입 중량'].sum() / 1000000
                    st.metric("총 수입량 (백만 kg)", f"{total_volume:,.2f}")
                with col2:
                    total_value = st.session_state.df_combined['수입 금액'].sum() / 1000000
                    st.metric("총 수입금액 (백만 $)", f"{total_value:,.2f}")
                with col3:
                    avg_unit_price = (st.session_state.df_combined['수입 금액'] / st.session_state.df_combined['수입 중량']).mean()
                    st.metric("평균 단가 ($/kg)", f"{avg_unit_price:,.2f}")

                # 그래프: 수입량, 수입금액, 검색량
                st.subheader("기간별 수입량 및 검색량 추이")
                fig1 = px.line(
                    st.session_state.df_combined, 
                    x="기간", 
                    y=["수입 중량", "검색량"], 
                    labels={"value": "수량 / 검색량", "variable": "지표"},
                    title="월별 수입량과 검색량 추이"
                )
                fig1.update_traces(hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}")
                st.plotly_chart(fig1, use_container_width=True)

                # 국가별 수입량/금액 그래프
                st.subheader("국가별 수입량 및 금액")
                df_country = st.session_state.df_imports[
                    st.session_state.df_imports['HS코드'].astype(str).isin(selected_codes)
                ].groupby('국가').agg({
                    '수입 중량': 'sum',
                    '수입 금액': 'sum'
                }).reset_index().sort_values(by='수입 중량', ascending=False)
                
                df_tds_country = st.session_state.df_tds[
                    st.session_state.df_tds['HS코드'].astype(str).isin(selected_codes)
                ].groupby('Origin Country').agg({
                    'Volume': 'sum',
                    'Value': 'sum'
                }).reset_index().rename(columns={'Origin Country': '국가', 'Volume': '수입 중량', 'Value': '수입 금액'})
                
                df_country = pd.concat([df_country, df_tds_country]).groupby('국가').sum().reset_index()
                df_country = df_country.sort_values(by='수입 중량', ascending=False)

                col1_bar, col2_bar = st.columns(2)
                with col1_bar:
                    fig_country_vol = px.bar(
                        df_country.head(10), 
                        x='국가', 
                        y='수입 중량', 
                        title='주요 수입국 (수입량 기준)',
                        labels={'수입 중량': '수입량 (kg)'}
                    )
                    st.plotly_chart(fig_country_vol, use_container_width=True)
                with col2_bar:
                    fig_country_val = px.bar(
                        df_country.sort_values(by='수입 금액', ascending=False).head(10), 
                        x='국가', 
                        y='수입 금액', 
                        title='주요 수입국 (수입금액 기준)',
                        labels={'수입 금액': '수입금액 ($)'}
                    )
                    st.plotly_chart(fig_country_val, use_container_width=True)
            else:
                st.warning("데이터 통합에 실패했습니다. 올바른 HS코드가 포함된 데이터를 선택했는지 확인해주세요.")

        with tab2:
            st.header("수요/가격 예측 모델 (간단한 회귀 모델)")
            if not st.session_state.df_combined.empty:
                df_model = st.session_state.df_combined.copy()
                
                df_model['검색량_lag1'] = df_model['검색량'].shift(1)
                df_model.dropna(inplace=True)

                if not df_model.empty:
                    model = LinearRegression()
                    X = df_model[['검색량_lag1']]
                    y = df_model['수입 중량']
                    
                    model.fit(X, y)
                    df_model['예측 수입 중량'] = model.predict(X)
                    
                    st.write("---")
                    st.subheader("미래 수입량 예측")
                    
                    last_search_volume = df_model['검색량'].iloc[-1]
                    predicted_volume = model.predict([[last_search_volume]])[0]
                    
                    st.success(f"다음 달 예상 수입량은 **{predicted_volume:,.0f} kg** 입니다.")
                    st.info("💡 **전략 인사이트**: 검색량이 수입량으로 이어지는 경향이 있습니다. 검색량 추이를 지속적으로 모니터링하여 미리 물량을 확보하세요.")
            
                    st.subheader("예측 모델 결과 시각화")
                    fig_pred = px.line(
                        df_model, 
                        x='기간', 
                        y=['수입 중량', '예측 수입 중량'],
                        title='실제 수입량 vs. 예측 수입량',
                        labels={'value': '수입량 (kg)', 'variable': '지표'}
                    )
                    fig_pred.update_traces(hovertemplate="%{x|%Y-%m}<br>%{y:,.0f}")
                    st.plotly_chart(fig_pred, use_container_width=True)
                else:
                    st.warning("데이터가 너무 적어 예측 모델을 실행할 수 없습니다. 더 많은 데이터를 업로드해주세요.")
            else:
                st.warning("데이터를 업로드하거나 구글 시트에서 읽어와 예측 모델을 활성화하세요. HS코드를 선택해주세요.")

        with tab3:
            st.header("데이터 상관관계 분석")
            if not st.session_state.df_combined.empty:
                corr_matrix = st.session_state.df_combined[['수입 중량', '수입 금액', '검색량']].corr()
                st.subheader("상관관계 행렬")
                st.dataframe(corr_matrix.style.background_gradient(cmap='Blues'), use_container_width=True)

                st.markdown(
                    """
                    - **상관계수 1**: 완벽한 양의 상관관계 (한 변수 증가 시 다른 변수도 증가)
                    - **상관계수 -1**: 완벽한 음의 상관관계 (한 변수 증가 시 다른 변수는 감소)
                    - **상관계수 0**: 상관관계 없음
                    """
                )
                
                st.write("---")
                st.subheader("산점도 시각화")
                fig_scatter = px.scatter(
                    st.session_state.df_combined,
                    x='검색량',
                    y='수입 중량',
                    trendline='ols',
                    title='검색량과 수입량의 상관관계',
                    labels={'검색량': '네이버 검색량', '수입 중량': '수입량 (kg)'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                st.info("💡 **인사이트**: 검색량과 수입량 간의 양의 상관관계가 보인다면, 검색량 증가는 미래의 수요 증가를 시사합니다. 이를 통해 수입 물량 결정에 참고할 수 있습니다.")
            else:
                st.warning("데이터를 업로드하거나 구글 시트에서 읽어와 상관관계 분석을 활성화하세요. HS코드를 선택해주세요.")

        with tab4:
            st.header("원본 데이터")
            st.subheader("관세청 데이터")
            st.dataframe(st.session_state.df_imports, use_container_width=True)
            st.subheader("네이버 데이터랩 검색량")
            st.dataframe(st.session_state.df_naver, use_container_width=True)
            st.subheader("TDS 데이터")
            st.dataframe(st.session_state.df_tds, use_container_width=True)
