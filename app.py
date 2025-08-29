import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
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

st.title("🧭 Compass : Data-Driven Direction")

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
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(credentials)
        return gc
    except Exception as e:
        st.error(f"구글 시트 인증 오류: {e}")
        return None
        
def normalize_hscode(hscode_series):
    """
    HS코드를 10자리 문자열로 정규화합니다.
    """
    return hscode_series.astype(str).str.strip().str.zfill(10)

def read_google_sheet(sheet_name):
    """
    지정된 구글 시트의 워크시트에서 데이터를 읽어 DataFrame으로 반환합니다.
    """
    gc = get_google_sheet_client()
    if gc:
        try:
            sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/12YdcKX3nvaNfFWYkJApRnoKAQnjCeR09AGRJ6rBiOuM/edit?gid=0#gid=0")
            worksheet = sh.worksheet(sheet_name)
            
            all_data = worksheet.get_all_values()
            if not all_data:
                return pd.DataFrame()
            
            headers = all_data[0]
            seen = {}
            for i, header in enumerate(headers):
                if not header:
                    headers[i] = f'Unnamed_{i}'
                elif header in seen:
                    headers[i] = f'{header}_{seen[header]}'
                    seen[header] += 1
                else:
                    seen[header] = 1

            data = all_data[1:]
            df = pd.DataFrame(data, columns=headers)
            
            # 데이터 정제 및 타입 변환
            if sheet_name == '네이버 데이터랩':
                if '커피' in df.columns:
                    df.rename(columns={'커피': '검색량'}, inplace=True)
                df['검색량'] = pd.to_numeric(df['검색량'], errors='coerce')
            elif sheet_name == 'TDS':
                if 'Detailed HS-CODE' in df.columns:
                    df.rename(columns={'Detailed HS-CODE': 'HS코드'}, inplace=True)
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                if 'HS코드' in df.columns:
                    df['HS코드'] = normalize_hscode(df['HS코드'])
            elif sheet_name == '관세청':
                df['수입 중량'] = pd.to_numeric(df['수입 중량'], errors='coerce')
                df['수입 금액'] = pd.to_numeric(df['수입 금액'], errors='coerce')
                if 'HS코드' in df.columns:
                    df['HS코드'] = normalize_hscode(df['HS코드'])

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
            # 수입량, 수입금액 숫자형 변환
            df['수입 중량'] = pd.to_numeric(df['수입 중량'], errors='coerce')
            df['수입 금액'] = pd.to_numeric(df['수입 금액'], errors='coerce')
            # HS코드 정규화
            if 'HS코드' in df.columns:
                df['HS코드'] = normalize_hscode(df['HS코드'])

            st.session_state.df_imports = pd.concat([st.session_state.df_imports, df], ignore_index=True)
            st.sidebar.success("관세청 데이터 업로드 완료!")
        except Exception as e:
            st.sidebar.error(f"관세청 CSV 파일 형식이 올바르지 않습니다: {e}")

    if uploaded_naver:
        try:
            # 네이버 데이터랩 CSV 파일은 첫 번째 행에 헤더가 있습니다.
            df = pd.read_csv(uploaded_naver, skiprows=6)
            
            # '커피'라는 컬럼을 '검색량'으로 변경합니다.
            if '커피' in df.columns:
                df.rename(columns={'커피': '검색량'}, inplace=True)
            
            # 검색량 컬럼을 숫자형으로 변환합니다.
            df['검색량'] = pd.to_numeric(df['검색량'], errors='coerce')
            
            st.session_state.df_naver = pd.concat([st.session_state.df_naver, df], ignore_index=True)
            st.sidebar.success("네이버 데이터랩 업로드 완료!")
        except Exception as e:
            st.sidebar.error(f"네이버 데이터랩 CSV 파일 형식이 올바르지 않습니다: {e}")

    if uploaded_tds:
        try:
            df_raw = uploaded_tds.getvalue().decode("utf-8")
            df = pd.read_csv(io.StringIO(df_raw), header=None)
            
            headers = df.iloc[0].tolist()
            seen = {}
            new_headers = []
            for i, header in enumerate(headers):
                if not isinstance(header, str) or not header:
                    new_header = f'Unnamed_{i}'
                elif header in seen:
                    seen[header] += 1
                    new_header = f'{header}_{seen[header]}'
                else:
                    seen[header] = 1
                    new_header = header
                new_headers.append(new_header)
            
            df.columns = new_headers
            df = df.iloc[1:].reset_index(drop=True)
            
            # TDS 데이터의 수치형 컬럼 변환
            if 'Detailed HS-CODE' in df.columns:
                df.rename(columns={'Detailed HS-CODE': 'HS코드'}, inplace=True)
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            if 'Value' in df.columns:
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            # HS코드 정규화
            if 'HS코드' in df.columns:
                df['HS코드'] = normalize_hscode(df['HS코드'])

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
    # HS코드 목록 생성 및 전처리
    df_imports_hscodes = st.session_state.df_imports[['HS코드', '품목명']].dropna()
    df_tds_hscodes = st.session_state.df_tds.rename(columns={'Product Description': '품목명'})[['HS코드', '품목명']].dropna()
    
    all_hscodes = pd.concat([df_imports_hscodes, df_tds_hscodes]).drop_duplicates(subset='HS코드').sort_values(by='HS코드').reset_index(drop=True)
    
    # 10자리 숫자 HS코드만 필터링
    all_hscodes = all_hscodes[all_hscodes['HS코드'].str.strip().str.len() == 10]
    all_hscodes['display_name'] = all_hscodes['HS코드'].astype(str) + ' - ' + all_hscodes['품목명']
    
    hscode_options = all_hscodes['display_name'].tolist()

    st.session_state.selected_hscodes = st.sidebar.multiselect(
        "분석할 HS코드를 선택하세요",
        options=hscode_options,
        default=hscode_options[:2] if len(hscode_options) > 1 else hscode_options
    )

    selected_codes = [s.split(' - ')[0] for s in st.session_state.selected_hscodes]

    if not selected_codes:
        st.warning("분석을 위해 최소 하나 이상의 HS코드를 선택해야 합니다.")
    else:
        # 기간 선택 슬라이더
        min_date_ts = pd.to_datetime('2020-01-01')
        max_date_ts = pd.to_datetime(datetime.now())
        start_date, end_date = st.sidebar.slider(
            "분석 기간을 선택하세요",
            min_value=min_date_ts.to_pydatetime(),
            max_value=max_date_ts.to_pydatetime(),
            value=(min_date_ts.to_pydatetime(), max_date_ts.to_pydatetime()),
            format="YYYY-MM-DD"
        )
        
        with st.spinner('데이터를 통합하는 중입니다...'):
            try:
                # 관세청과 TDS 데이터 통합
                df_imports_filtered = st.session_state.df_imports[
                    st.session_state.df_imports['HS코드'].astype(str).isin(selected_codes)
                ].copy()
                df_imports_filtered['기간'] = pd.to_datetime(df_imports_filtered['기간'], format='%Y.%m', errors='coerce')
                
                df_tds_filtered = st.session_state.df_tds[
                    st.session_state.df_tds['HS코드'].astype(str).isin(selected_codes)
                ].copy()
                df_tds_filtered['Date'] = pd.to_datetime(df_tds_filtered['Date'], errors='coerce')
                
                # '기간'과 'Date' 열을 기준으로 데이터 병합
                df_combined_imports_tds = pd.concat([
                    df_imports_filtered.rename(columns={'수입 중량': 'Volume', '수입 금액': 'Value', '기간': 'Date', '국가': 'Origin Country'}),
                    df_tds_filtered
                ], ignore_index=True)
                
                # 기간 필터링
                df_combined_imports_tds['Date'] = pd.to_datetime(df_combined_imports_tds['Date'], errors='coerce')
                df_combined_imports_tds.dropna(subset=['Date'], inplace=True)
                df_combined_imports_tds = df_combined_imports_tds[
                    (df_combined_imports_tds['Date'] >= pd.Timestamp(start_date)) & 
                    (df_combined_imports_tds['Date'] <= pd.Timestamp(end_date))
                ]
                
                # 월별로 데이터 그룹화
                df_combined_monthly = df_combined_imports_tds.groupby(
                    pd.Grouper(key='Date', freq='M')
                ).agg({
                    'Volume': 'sum',
                    'Value': 'sum'
                }).reset_index().rename(columns={'Volume': '수입 중량', 'Value': '수입 금액'})
                
                # 네이버 데이터랩 전처리
                df_naver_monthly = st.session_state.df_naver.copy()
                df_naver_monthly['날짜'] = pd.to_datetime(df_naver_monthly['날짜'], errors='coerce')
                df_naver_monthly.dropna(subset=['날짜'], inplace=True)
                df_naver_monthly = df_naver_monthly[
                    (df_naver_monthly['날짜'] >= pd.Timestamp(start_date)) & 
                    (df_naver_monthly['날짜'] <= pd.Timestamp(end_date))
                ]
                df_naver_monthly = df_naver_monthly.groupby(
                    pd.Grouper(key='날짜', freq='M')
                ).agg({'검색량': 'mean'}).reset_index()

                # 최종 데이터 통합
                df_combined = pd.merge(
                    df_combined_monthly,
                    df_naver_monthly,
                    left_on=df_combined_monthly['Date'].dt.strftime('%Y-%m'),
                    right_on=df_naver_monthly['날짜'].dt.strftime('%Y-%m'),
                    how='outer'
                )
                
                # 열 이름 및 결측치 정리
                df_combined.rename(columns={'key_0': '기간'}, inplace=True)
                df_combined.drop(['Date', '날짜'], axis=1, errors='ignore', inplace=True)
                df_combined['수입 중량'].fillna(0, inplace=True)
                df_combined['수입 금액'].fillna(0, inplace=True)
                df_combined['검색량'].fillna(0, inplace=True)
                
                # TDS에서 온 '기간' 컬럼이 있으면 제거
                if '기간_y' in df_combined.columns:
                    df_combined.drop('기간_y', axis=1, inplace=True)

                st.session_state.df_combined = df_combined
                st.success("데이터 통합 완료!")
            except Exception as e:
                st.error(f"데이터 통합 중 오류가 발생했습니다. 선택한 HS코드에 해당하는 데이터가 없거나, 업로드한 파일 형식을 확인해주세요: {e}")

        # -----------------
        # 탭 구성
        # -----------------
        tab1, tab2, tab3, tab4 = st.tabs(["📊 대시보드", "🔮 예측 모델", "📈 상관관계 분석", "🗃️ 원본 데이터"])

        with tab1:
            st.header("커피 원두 시장 동향 분석")
            if not st.session_state.df_combined.empty and not st.session_state.df_combined['수입 중량'].sum() == 0:
                # KPI 지표
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_volume = st.session_state.df_combined['수입 중량'].sum() / 1000000
                    st.metric("총 수입량 (백만 kg)", f"{total_volume:,.2f}")
                with col2:
                    total_value = st.session_state.df_combined['수입 금액'].sum() / 1000000
                    st.metric("총 수입금액 (백만 $)", f"{total_value:,.2f}")
                with col3:
                    valid_data = st.session_state.df_combined[st.session_state.df_combined['수입 중량'] > 0]
                    avg_unit_price = (valid_data['수입 금액'] / valid_data['수입 중량']).mean()
                    st.metric("평균 단가 ($/kg)", f"{avg_unit_price:,.2f}" if not pd.isna(avg_unit_price) else "N/A")

                # 그래프: 수입량, 수입금액, 검색량
                st.subheader("기간별 수입량 및 검색량 추이")
                
                fig1 = make_subplots(specs=[[{"secondary_y": True}]])

                # 수입량 그래프
                fig1.add_trace(
                    go.Scatter(
                        x=st.session_state.df_combined['기간'], 
                        y=st.session_state.df_combined['수입 중량'], 
                        name='수입 중량'
                    ),
                    secondary_y=False,
                )

                # 검색량 그래프
                fig1.add_trace(
                    go.Scatter(
                        x=st.session_state.df_combined['기간'], 
                        y=st.session_state.df_combined['검색량'], 
                        name='검색량'
                    ),
                    secondary_y=True,
                )

                # 레이아웃 업데이트
                fig1.update_layout(
                    title_text="월별 수입량과 검색량 추이",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                # Y축 라벨 설정
                fig1.update_yaxes(title_text="<b>수입량 (kg)</b>", secondary_y=False)
                fig1.update_yaxes(title_text="<b>검색량</b>", secondary_y=True)

                st.plotly_chart(fig1, use_container_width=True)

                # -------------------------
                # 원산지별 가격 경쟁력 및 공급 안정성 분석
                # -------------------------
                st.subheader("원산지별 가격 경쟁력 및 공급 안정성 분석")
                
                # TDS와 관세청 데이터 통합
                df_country_analysis = df_combined_imports_tds.copy()
                df_country_analysis.rename(columns={'Volume': '수입 중량', 'Value': '수입 금액'}, inplace=True)
                df_country_analysis['단가'] = df_country_analysis['수입 금액'] / df_country_analysis['수입 중량']
                df_country_analysis.dropna(subset=['단가', 'Origin Country'], inplace=True)

                if not df_country_analysis.empty:
                    col_price, col_stability = st.columns(2)

                    with col_price:
                        # 가격 경쟁력 (평균 단가)
                        price_competitiveness = df_country_analysis.groupby('Origin Country')['단가'].mean().reset_index()
                        price_competitiveness = price_competitiveness.sort_values(by='단가', ascending=True)
                        fig_price = px.bar(
                            price_competitiveness.head(10),
                            x='Origin Country',
                            y='단가',
                            title='평균 단가($/kg) - 가격 경쟁력 (낮을수록 유리)',
                            labels={'단가': '평균 단가 ($/kg)', 'Origin Country': '원산지'}
                        )
                        st.plotly_chart(fig_price, use_container_width=True)

                    with col_stability:
                        # 공급 안정성 (수입 중량 변동성)
                        monthly_volume = df_country_analysis.groupby([pd.Grouper(key='Date', freq='M'), 'Origin Country'])['수입 중량'].sum().reset_index()
                        stability = monthly_volume.groupby('Origin Country')['수입 중량'].std().reset_index().rename(columns={'수입 중량': '변동성'})
                        stability = stability.sort_values(by='변동성', ascending=True)
                        fig_stability = px.bar(
                            stability.head(10),
                            x='Origin Country',
                            y='변동성',
                            title='공급량 변동성 (낮을수록 안정적)',
                            labels={'변동성': '표준편차', 'Origin Country': '원산지'}
                        )
                        st.plotly_chart(fig_stability, use_container_width=True)
                else:
                    st.warning("선택한 HS코드에 대한 원산지별 데이터가 충분하지 않아 분석을 표시할 수 없습니다.")
                
                # 국가별 수입량/금액 그래프
                st.subheader("국가별 수입량 및 금액")
                
                df_imports_country = st.session_state.df_imports[
                    st.session_state.df_imports['HS코드'].astype(str).isin(selected_codes)
                ].groupby('국가').agg({
                    '수입 중량': 'sum',
                    '수입 금액': 'sum'
                }).reset_index()
                
                df_tds_country = st.session_state.df_tds[
                    st.session_state.df_tds['HS코드'].astype(str).isin(selected_codes)
                ].groupby('Origin Country').agg({
                    'Volume': 'sum',
                    'Value': 'sum'
                }).reset_index().rename(columns={'Origin Country': '국가', 'Volume': '수입 중량', 'Value': '수입 금액'})
                
                df_country = pd.concat([df_imports_country, df_tds_country]).groupby('국가').sum().reset_index()
                df_country = df_country.sort_values(by='수입 중량', ascending=False)
                
                if not df_country.empty:
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
                        st.plotly_chart(fig_country_val, use_container_chart=True)
                else:
                    st.warning("선택한 HS코드에 대한 국가별 데이터가 없습니다.")
            else:
                st.warning("선택한 HS코드에 대한 데이터가 존재하지 않아 대시보드를 표시할 수 없습니다.")

        with tab2:
            st.header("수요/가격 예측 모델 (간단한 회귀 모델)")
            st.markdown("""
            ---
            ### **예측 로직 설명**
            이 모델은 **단순 선형 회귀(Linear Regression)**를 사용합니다. 과거 **'네이버 검색량'** 데이터가 다음 달 **'수입 중량'**에 미치는 영향을 분석하여 미래의 수입량을 예측합니다. 즉, 소비자의 검색 관심도(수요)가 실제 수입(공급)으로 이어지는 경향을 파악하여 예측하는 방식입니다.

            **💡 전략 인사이트**: 검색량이 수입량으로 이어지는 경향이 있습니다. 검색량 추이를 지속적으로 모니터링하여 미리 물량을 확보하면 재고 및 공급망 관리에 유리합니다.
            """)
            
            if not st.session_state.df_combined.empty and not st.session_state.df_combined['수입 중량'].sum() == 0:
                df_model = st.session_state.df_combined.copy()
                
                df_model['검색량_lag1'] = df_model['검색량'].shift(1)
                df_model.dropna(inplace=True)

                if not df_model.empty:
                    # statsmodels를 사용하여 신뢰구간을 계산합니다.
                    X = sm.add_constant(df_model['검색량_lag1'])
                    y = df_model['수입 중량']
                    
                    model = sm.OLS(y, X).fit()
                    
                    # 예측값과 95% 신뢰구간을 가져옵니다.
                    predictions = model.get_prediction(X)
                    df_model['예측 수입 중량'] = predictions.predicted_mean
                    conf_int = predictions.conf_int(alpha=0.05)
                    df_model['conf_int_lower'] = conf_int[:, 0]
                    df_model['conf_int_upper'] = conf_int[:, 1]
                    
                    st.write("---")
                    st.subheader("미래 수입량 예측")
                    
                    last_search_volume = df_model['검색량'].iloc[-1]
                    predicted_volume = model.predict([1, last_search_volume])[0]
                    
                    st.success(f"다음 달 예상 수입량은 **{predicted_volume:,.0f} kg** 입니다.")
                    st.info("💡 **전략 인사이트**: 검색량이 수입량으로 이어지는 경향이 있습니다. 검색량 추이를 지속적으로 모니터링하여 미리 물량을 확보하세요.")
            
                    st.subheader("예측 모델 결과 시각화")
                    fig_pred = go.Figure()

                    # 실제 수입량
                    fig_pred.add_trace(go.Scatter(
                        x=df_model['기간'],
                        y=df_model['수입 중량'],
                        mode='lines',
                        name='실제 수입 중량'
                    ))

                    # 예측 수입량
                    fig_pred.add_trace(go.Scatter(
                        x=df_model['기간'],
                        y=df_model['예측 수입 중량'],
                        mode='lines',
                        name='예측 수입 중량',
                        line=dict(color='red', dash='dash')
                    ))

                    # 신뢰구간 (음영)
                    fig_pred.add_trace(go.Scatter(
                        x=df_model['기간'],
                        y=df_model['conf_int_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=df_model['기간'],
                        y=df_model['conf_int_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(200, 200, 200, 0.2)',
                        name='95% 신뢰구간'
                    ))

                    fig_pred.update_layout(
                        title_text="실제 수입량 vs. 예측 수입량 (95% 신뢰구간)",
                        xaxis_title="기간",
                        yaxis_title="수입량 (kg)"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                else:
                    st.warning("데이터가 너무 적어 예측 모델을 실행할 수 없습니다. 더 많은 데이터를 업로드해주세요.")
            else:
                st.warning("선택한 HS코드에 대한 데이터가 존재하지 않아 예측 모델을 활성화할 수 없습니다.")

        with tab3:
            st.header("데이터 상관관계 분석")
            if not st.session_state.df_combined.empty and not st.session_state.df_combined['수입 중량'].sum() == 0:
                corr_matrix = st.session_state.df_combined[['수입 중량', '수입 금액', '검색량']].corr()
                st.subheader("상관관계 행렬")
                st.dataframe(corr_matrix, use_container_width=True)

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
                st.warning("선택한 HS코드에 대한 데이터가 존재하지 않아 상관관계 분석을 활성화할 수 없습니다.")

        with tab4:
            st.header("원본 데이터")
            st.subheader("관세청 데이터")
            st.dataframe(st.session_state.df_imports, use_container_width=True)
            st.subheader("네이버 데이터랩 검색량")
            st.dataframe(st.session_state.df_naver, use_container_width=True)
            st.subheader("TDS 데이터")
            st.dataframe(st.session_state.df_tds, use_container_width=True)
