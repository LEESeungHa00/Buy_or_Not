import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import io
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

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
if 'top_countries' not in st.session_state:
    st.session_state.top_countries = []

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
                df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
            elif sheet_name == 'TDS':
                if 'Detailed HS-CODE' in df.columns:
                    df.rename(columns={'Detailed HS-CODE': 'HS코드'}, inplace=True)
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                if 'HS코드' in df.columns:
                    df['HS코드'] = normalize_hscode(df['HS코드'])
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # TDS에만 있는 컬럼 확인 및 추가
                if 'Raw Importer Name' not in df.columns: df['Raw Importer Name'] = ''
                if 'Exporter' not in df.columns: df['Exporter'] = ''
            elif sheet_name == '관세청':
                df['수입 중량'] = pd.to_numeric(df['수입 중량'], errors='coerce')
                df['수입 금액'] = pd.to_numeric(df['수입 금액'], errors='coerce')
                if 'HS코드' in df.columns:
                    df['HS코드'] = normalize_hscode(df['HS코드'])
                if '기간' in df.columns:
                    df['기간'] = pd.to_datetime(df['기간'], errors='coerce')
                    df.rename(columns={'기간': 'Date'}, inplace=True)

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
            df['기간'] = pd.to_datetime(df['기간'], errors='coerce')
            df.rename(columns={'기간': 'Date'}, inplace=True)
            df['수입 중량'] = pd.to_numeric(df['수입 중량'], errors='coerce')
            df['수입 금액'] = pd.to_numeric(df['수입 금액'], errors='coerce')
            if 'HS코드' in df.columns:
                df['HS코드'] = normalize_hscode(df['HS코드'])

            st.session_state.df_imports = pd.concat([st.session_state.df_imports, df], ignore_index=True)
            st.sidebar.success("관세청 데이터 업로드 완료!")
        except Exception as e:
            st.sidebar.error(f"관세청 CSV 파일 형식이 올바르지 않습니다: {e}")

    if uploaded_naver:
        try:
            df = pd.read_csv(uploaded_naver, skiprows=6)
            
            if '커피' in df.columns:
                df.rename(columns={'커피': '검색량'}, inplace=True)
            
            df['검색량'] = pd.to_numeric(df['검색량'], errors='coerce')
            df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
            
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
            
            if 'Detailed HS-CODE' in df.columns:
                df.rename(columns={'Detailed HS-CODE': 'HS코드'}, inplace=True)
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            if 'Value' in df.columns:
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            if 'HS코드' in df.columns:
                df['HS코드'] = normalize_hscode(df['HS코드'])
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # TDS에만 있는 컬럼 확인 및 추가
            if 'Raw Importer Name' not in df.columns: df['Raw Importer Name'] = ''
            if 'Exporter' not in df.columns: df['Exporter'] = ''

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
    df_imports_renamed = st.session_state.df_imports.rename(columns={'국가': 'Origin Country', '수입 중량': 'Volume', '수입 금액': 'Value'})
    
    # Ensure TDS-specific columns are present in df_imports_renamed
    if 'Raw Importer Name' not in df_imports_renamed.columns: df_imports_renamed['Raw Importer Name'] = ''
    if 'Exporter' not in df_imports_renamed.columns: df_imports_renamed['Exporter'] = ''

    df_combined_imports_tds = pd.concat([
        df_imports_renamed,
        st.session_state.df_tds.rename(columns={'Product Description': '품목명'})
    ], ignore_index=True)
    
    df_combined_imports_tds = df_combined_imports_tds[
        (df_combined_imports_tds['Volume'] > 0) &
        (df_combined_imports_tds['Value'] > 0) &
        (df_combined_imports_tds['HS코드'].notna()) &
        (df_combined_imports_tds['Origin Country'].notna())
    ].copy()

    all_hscodes = df_combined_imports_tds[['HS코드', '품목명']].dropna().drop_duplicates(subset='HS코드').sort_values(by='HS코드').reset_index(drop=True)
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
        df_filtered = df_combined_imports_tds[
            df_combined_imports_tds['HS코드'].astype(str).isin(selected_codes)
        ].copy()

        min_date_ts = pd.to_datetime(df_filtered['Date'].min())
        max_date_ts = pd.to_datetime(df_filtered['Date'].max())
        start_date, end_date = st.sidebar.slider(
            "분석 기간을 선택하세요",
            min_value=min_date_ts.to_pydatetime(),
            max_value=max_date_ts.to_pydatetime(),
            value=(min_date_ts.to_pydatetime(), max_date_ts.to_pydatetime()),
            format="YYYY-MM-DD"
        )
        
        with st.spinner('데이터를 통합하는 중입니다...'):
            try:
                df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
                df_filtered.dropna(subset=['Date'], inplace=True)
                df_filtered = df_filtered[
                    (df_filtered['Date'] >= pd.Timestamp(start_date)) & 
                    (df_filtered['Date'] <= pd.Timestamp(end_date))
                ]
                
                df_combined_monthly = df_filtered.groupby(
                    pd.Grouper(key='Date', freq='M')
                ).agg({
                    'Volume': 'sum',
                    'Value': 'sum'
                }).reset_index().rename(columns={'Volume': '수입 중량', 'Value': '수입 금액'})
                
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

                df_combined = pd.merge(
                    df_combined_monthly,
                    df_naver_monthly,
                    left_on=df_combined_monthly['Date'].dt.strftime('%Y-%m'),
                    right_on=df_naver_monthly['날짜'].dt.strftime('%Y-%m'),
                    how='outer'
                )
                
                df_combined.rename(columns={'key_0': '기간'}, inplace=True)
                df_combined.drop(['Date', '날짜'], axis=1, errors='ignore', inplace=True)
                df_combined['수입 중량'].fillna(0, inplace=True)
                df_combined['수입 금액'].fillna(0, inplace=True)
                df_combined['검색량'].fillna(0, inplace=True)
                
                if '기간_y' in df_combined.columns:
                    df_combined.drop('기간_y', axis=1, inplace=True)

                st.session_state.df_combined = df_combined
                st.success("데이터 통합 완료!")
            except Exception as e:
                st.error(f"데이터 통합 중 오류가 발생했습니다. 업로드한 파일 형식을 확인해주세요: {e}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 대시보드", "🔮 예측 모델", "📈 상관관계 분석", "🗺️ 공급망 분석", "🗃️ 원본 데이터"])

        with tab1:
            st.header("커피 원두 시장 동향 분석")
            total_filtered_rows = df_filtered.shape[0]
            st.info(f"선택한 HS코드에 대한 총 데이터 행: {total_filtered_rows}개")

            if not st.session_state.df_combined.empty and not st.session_state.df_combined['수입 중량'].sum() == 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_volume = st.session_state.df_combined['수입 중량'].sum() / 1000000
                    st.metric("총 수입량 (백만 kg)", f"{total_volume:,.2f}")
                with col2:
                    total_value = st.session_state.df_combined['수입 금액'].sum() / 1000000
                    st.metric("총 수입금액 (백만 $)", f"{total_value:,.2f}")
                with col3:
                    valid_data = df_filtered[df_filtered['Volume'] > 0]
                    avg_unit_price = (valid_data['Value'] / valid_data['Volume']).mean()
                    st.metric("평균 단가 ($/kg)", f"{avg_unit_price:,.2f}" if not pd.isna(avg_unit_price) else "N/A")

                st.subheader("기간별 수입량 및 검색량 추이")
                
                fig1 = make_subplots(specs=[[{"secondary_y": True}]])

                fig1.add_trace(
                    go.Scatter(
                        x=st.session_state.df_combined['기간'], 
                        y=st.session_state.df_combined['수입 중량'], 
                        name='수입 중량'
                    ),
                    secondary_y=False,
                )

                fig1.add_trace(
                    go.Scatter(
                        x=st.session_state.df_combined['기간'], 
                        y=st.session_state.df_combined['검색량'], 
                        name='검색량'
                    ),
                    secondary_y=True,
                )

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

                fig1.update_yaxes(title_text="<b>수입량 (kg)</b>", secondary_y=False)
                fig1.update_yaxes(title_text="<b>검색량</b>", secondary_y=True)

                st.plotly_chart(fig1, use_container_width=True)

                st.subheader("원산지별 가격 경쟁력 및 공급 안정성 분석")
                
                top_10_countries = df_filtered.groupby('Origin Country')['Volume'].sum().nlargest(10).index.tolist()
                df_country_analysis = df_filtered[df_filtered['Origin Country'].isin(top_10_countries)].copy()
                df_country_analysis['단가'] = df_country_analysis['Value'] / df_country_analysis['Volume']
                df_country_analysis.dropna(subset=['단가', 'Origin Country'], inplace=True)
                
                if 'Exporter' in df_filtered.columns and 'Raw Importer Name' in df_filtered.columns:
                    df_filtered['Exporter'].fillna('', inplace=True)
                    df_filtered['Raw Importer Name'].fillna('', inplace=True)

                if not df_country_analysis.empty:
                    col_price, col_stability = st.columns(2)

                    with col_price:
                        price_competitiveness = df_country_analysis.groupby('Origin Country')['단가'].mean().reset_index()
                        price_competitiveness = price_competitiveness.sort_values(by='단가', ascending=True)
                        fig_price = go.Figure(px.bar(
                            price_competitiveness,
                            x='Origin Country',
                            y='단가',
                            title='평균 단가($/kg) - 가격 경쟁력 (낮을수록 유리)',
                            labels={'단가': '평균 단가 ($/kg)', 'Origin Country': '원산지'}
                        ))
                        fig_price.update_traces(hovertemplate='<b>%{x}</b><br>평균 단가: %{y:,.2f} $/kg<extra></extra>')
                        fig_price.update_layout(xaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_price, use_container_width=True)

                    with col_stability:
                        monthly_volume = df_country_analysis.groupby([pd.Grouper(key='Date', freq='M'), 'Origin Country'])['Volume'].sum().reset_index()
                        stability = monthly_volume.groupby('Origin Country')['Volume'].std().reset_index().rename(columns={'Volume': '변동성'})
                        stability = stability.sort_values(by='변동성', ascending=True)
                        fig_stability = go.Figure(px.bar(
                            stability,
                            x='Origin Country',
                            y='변동성',
                            title='공급량 변동성 (낮을수록 안정적)',
                            labels={'변동성': '표준편차', 'Origin Country': '원산지'}
                        ))
                        fig_stability.update_traces(hovertemplate='<b>%{x}</b><br>공급량 변동성: %{y:,.2f}<extra></extra>')
                        fig_stability.update_layout(xaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_stability, use_container_width=True)
                else:
                    st.warning("선택한 HS코드에 대한 원산지별 데이터가 충분하지 않아 분석을 표시할 수 없습니다.")
                
                st.subheader("국가별 수입량 및 금액")
                
                df_country_filtered = df_filtered[df_filtered['Origin Country'].isin(top_10_countries)]
                
                df_country = df_country_filtered.groupby('Origin Country').agg({
                    'Volume': 'sum',
                    'Value': 'sum'
                }).reset_index().rename(columns={'Origin Country': '국가', 'Volume': '수입 중량', 'Value': '수입 금액'})

                df_country = df_country.sort_values(by='수입 중량', ascending=False)
                
                if not df_country.empty:
                    col1_bar, col2_bar = st.columns(2)
                    with col1_bar:
                        fig_country_vol = go.Figure(px.bar(
                            df_country.head(10), 
                            x='국가', 
                            y='수입 중량', 
                            title='주요 수입국 (수입량 기준)',
                            labels={'수입 중량': '수입량 (kg)'}
                        ))
                        fig_country_vol.update_traces(hovertemplate='<b>%{x}</b><br>수입 중량: %{y:,.0f} kg<extra></extra>', text=None, texttemplate=None)
                        st.plotly_chart(fig_country_vol, use_container_width=True)
                    with col2_bar:
                        fig_country_val = go.Figure(px.bar(
                            df_country.sort_values(by='수입 금액', ascending=False).head(10), 
                            x='국가', 
                            y='수입 금액', 
                            title='주요 수입국 (수입금액 기준)',
                            labels={'수입 금액': '수입금액 ($)'}
                        ))
                        fig_country_val.update_traces(hovertemplate='<b>%{x}</b><br>수입 금액: %{y:,.0f} $<extra></extra>', text=None, texttemplate=None)
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
                    X = sm.add_constant(df_model['검색량_lag1'])
                    y = df_model['수입 중량']
                    
                    model = sm.OLS(y, X).fit()
                    
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

                    fig_pred.add_trace(go.Scatter(
                        x=df_model['기간'],
                        y=df_model['수입 중량'],
                        mode='lines',
                        name='실제 수입 중량'
                    ))

                    fig_pred.add_trace(go.Scatter(
                        x=df_model['기간'],
                        y=df_model['예측 수입 중량'],
                        mode='lines',
                        name='예측 수입 중량',
                        line=dict(color='red', dash='dash')
                    ))

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
            st.header("공급망 분석")
            
            if not df_filtered.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("원산지별 수입량 비중")
                    df_pie = df_filtered.groupby('Origin Country')['Volume'].sum().reset_index()
                    fig_pie = px.pie(
                        df_pie, 
                        values='Volume', 
                        names='Origin Country', 
                        title='총 수입 중량 비중',
                        labels={'Volume': '수입 중량'}
                    )
                    fig_pie.update_traces(textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    st.subheader("핵심 원산지 수입량 추이")
                    top_2_countries = df_filtered.groupby('Origin Country')['Volume'].sum().nlargest(2).index.tolist()
                    if len(top_2_countries) > 1:
                        df_top2 = df_filtered[df_filtered['Origin Country'].isin(top_2_countries)].copy()
                        df_top2_monthly = df_top2.groupby([
                            pd.Grouper(key='Date', freq='M'), 'Origin Country'
                        ])['Volume'].sum().reset_index()
                        
                        fig_top2 = px.line(
                            df_top2_monthly, 
                            x='Date', 
                            y='Volume', 
                            color='Origin Country',
                            title=f"{top_2_countries[0]} vs. {top_2_countries[1]} 수입량 추이",
                            labels={'Volume': '수입 중량 (kg)', 'Date': '기간'}
                        )
                        st.plotly_chart(fig_top2, use_container_width=True)
                    else:
                        st.warning("분석할 상위 2개 국가 데이터가 충분하지 않습니다.")
                
                st.subheader("수입/수출업체 현황 분석")
                
                # Check for the existence of the TDS-specific columns and handle them gracefully
                if 'Raw Importer Name' in df_filtered.columns and 'Exporter' in df_filtered.columns and df_filtered['Raw Importer Name'].any() and df_filtered['Exporter'].any():
                    col_importer, col_exporter = st.columns(2)
                    
                    with col_importer:
                        st.markdown("### 주요 수입업체")
                        importers_by_volume = df_filtered.groupby('Raw Importer Name')['Volume'].sum().nlargest(10).reset_index()
                        fig_importer = go.Figure(px.bar(
                            importers_by_volume,
                            x='Raw Importer Name',
                            y='Volume',
                            title='수입량 기준 상위 10개 수입업체',
                            labels={'Raw Importer Name': '수입업체', 'Volume': '수입 중량 (kg)'}
                        ))
                        fig_importer.update_traces(hovertemplate='<b>%{x}</b><br>수입 중량: %{y:,.0f} kg<extra></extra>')
                        st.plotly_chart(fig_importer, use_container_width=True)

                    with col_exporter:
                        st.markdown("### 주요 수출업체")
                        exporters_by_volume = df_filtered.groupby('Exporter')['Volume'].sum().nlargest(10).reset_index()
                        fig_exporter = go.Figure(px.bar(
                            exporters_by_volume,
                            x='Exporter',
                            y='Volume',
                            title='수입량 기준 상위 10개 수출업체',
                            labels={'Exporter': '수출업체', 'Volume': '수입 중량 (kg)'}
                        ))
                        fig_exporter.update_traces(hovertemplate='<b>%{x}</b><br>수입 중량: %{y:,.0f} kg<extra></extra>')
                        st.plotly_chart(fig_exporter, use_container_width=True)

                    st.subheader("특정 원산지별 업체 분석")
                    
                    all_countries = df_filtered['Origin Country'].dropna().unique().tolist()
                    selected_country_importer = st.selectbox(
                        "업체 현황을 분석할 원산지를 선택하세요",
                        options=all_countries
                    )
                    
                    if selected_country_importer:
                        df_country_importers = df_filtered[df_filtered['Origin Country'] == selected_country_importer].copy()
                        if not df_country_importers.empty:
                            df_importers_ranked = df_country_importers.groupby('Raw Importer Name')['Volume'].sum().nlargest(10).reset_index()
                            
                            st.markdown(f"**{selected_country_importer}에서 가장 많이 수입하는 업체**")
                            fig_country_importers = go.Figure(px.bar(
                                df_importers_ranked,
                                x='Raw Importer Name',
                                y='Volume',
                                title=f"{selected_country_importer} 수입량 기준 상위 10개 업체",
                                labels={'Raw Importer Name': '수입업체', 'Volume': '수입 중량 (kg)'}
                            ))
                            fig_country_importers.update_traces(hovertemplate='<b>%{x}</b><br>수입 중량: %{y:,.0f} kg<extra></extra>')
                            st.plotly_chart(fig_country_importers, use_container_width=True)

                            all_importers = df_filtered['Raw Importer Name'].dropna().unique().tolist()
                            country_importers = df_country_importers['Raw Importer Name'].dropna().unique().tolist()
                            other_importers = [imp for imp in all_importers if imp not in country_importers]
                            
                            df_other_importers = df_filtered[df_filtered['Raw Importer Name'].isin(other_importers)].copy()
                            df_other_importers_ranked = df_other_importers.groupby('Raw Importer Name')['Volume'].sum().nlargest(10).reset_index()
                            
                            if not df_other_importers_ranked.empty:
                                st.markdown(f"**{selected_country_importer} 외 다른 국가에서 많이 수입하는 업체**")
                                fig_other_importers = go.Figure(px.bar(
                                    df_other_importers_ranked,
                                    x='Raw Importer Name',
                                    y='Volume',
                                    title=f"{selected_country_importer} 외 수입량 기준 상위 10개 업체",
                                    labels={'Raw Importer Name': '수입업체', 'Volume': '수입 중량 (kg)'}
                                ))
                                fig_other_importers.update_traces(hovertemplate='<b>%{x}</b><br>수입 중량: %{y:,.0f} kg<extra></extra>')
                                st.plotly_chart(fig_other_importers, use_container_width=True)
                            else:
                                st.warning(f"다른 국가에서 수입하는 업체 정보가 없습니다.")
                        else:
                            st.warning(f"선택한 {selected_country_importer}에 대한 업체 데이터가 충분하지 않습니다.")
                else:
                    st.warning("TDS 데이터에 '수출업체' 또는 '수입업체' 정보가 없어 분석할 수 없습니다.")
            else:
                st.warning("선택한 HS코드에 대한 데이터가 존재하지 않아 공급망 분석을 활성화할 수 없습니다.")

        with tab5:
            st.header("원본 데이터")
            st.subheader("관세청 데이터")
            st.dataframe(st.session_state.df_imports, use_container_width=True)
            st.subheader("네이버 데이터랩 검색량")
            st.dataframe(st.session_state.df_naver, use_container_width=True)
            st.subheader("TDS 데이터")
            st.dataframe(st.session_state.df_tds, use_container_width=True)
