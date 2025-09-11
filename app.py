import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from kamis_data import KAMIS_FULL_DATA
import yfinance as yf
import itertools

# Advanced Analysis Libraries
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Transformers / HuggingFace
from transformers import pipeline
from huggingface_hub import login as hf_login

# Optional: robust article text fetch
try:
    from newspaper import Article, Config as NewsConfig
    _HAS_NEWSPAPER = True
except ImportError:
    _HAS_NEWSPAPER = False

# ----------------------------
#  Configuration / Constants
# ----------------------------
st.set_page_config(layout="wide")
st.title("📊 통합 데이터 기반 탐색 및 예측 대시보드")

BQ_DATASET = "data_explorer"
BQ_TABLE_TRADE = "tds_data"
BQ_TABLE_NAVER = "naver_trends_cache"
BQ_TABLE_NEWS = "news_sentiment_finbert"

COMMODITY_TICKERS = {
    "Coffee": "KC=F", "Cocoa": "CC=F", "Orange Juice": "OJ=F",
    "Crude Oil (WTI)": "CL=F", "Gold": "GC=F", "Corn": "ZC=F",
    "Soybeans": "ZS=F", "Wheat": "ZW=F", "Natural Gas": "NG=F", "Copper": "HG=F",
}

DEFAULT_MODEL_IDS = {
    "finbert": "snunlp/KR-FinBERT-SC",
    "elite": "nlptown/bert-base-multilingual-uncased-sentiment",
    "product": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
}

# --- Guide Content ---
guide_content = """
### 1. 상관관계 분석 탭
- **분석 방법 선택 (Pearson vs Spearman)**
    - **`Pearson`**: 두 변수가 **'선형 관계'(정비례/반비례)**에 가까울 때 선택합니다. (예: 광고비와 매출)
    - **`Spearman`**: **'순위 관계'(한쪽이 증가할 때 다른 쪽도 일관되게 증가/감소)**를 보고 싶을 때 선택합니다. 복잡한 비선형 관계도 잡아낼 수 있어 더 범용적입니다.
- **P-value 필터**: 이 관계가 **'우연일 확률'**을 의미합니다. 통계적으로 의미 있는 관계를 찾기 위해 보통 **0.05 이하**로 설정합니다. (즉, 우연일 확률이 5% 미만인 관계만 보겠다는 의미)
- **시차 분석 (선행/후행 변수)**
    - **`선행 변수(Driver)`**: 원인이 될 수 있는 변수 그룹 (예: 뉴스 감성, 선물 가격)
    - **`후행 변수(Outcome)`**: 결과적으로 영향을 받는 변수 그룹 (예: 수입량, 수입단가)
    - **목적**: "국제 유가가 오르면, 몇 주 후에 수입 단가가 상승할까?" 와 같은 선후 관계를 분석하여 미래 예측의 단서를 찾습니다.
- **산점도 행렬**: 여러 변수 간의 관계를 한눈에 시각적으로 파악할 수 있습니다. 점들이 우상향하면 양의 관계, 우하향하면 음의 관계를 의심해볼 수 있습니다.

---

### 2. 시계열 예측 탭
#### **가. 예측 파라미터, 어떻게 설정해야 할까요?**
- **`🤖 최적 파라미터 자동 탐색`**: 가장 먼저 눌러보세요! 데이터에 가장 적합한 파라미터 조합을 자동으로 찾아 슬라이더와 선택 박스를 설정해주는 기능입니다. 이 최적의 상태에서부터 미세 조정을 시작하는 것을 권장합니다.
- **`Trend 유연성 (changepoint_prior_scale)`**
    - **역할**: 데이터의 장기적인 추세(Trend)가 얼마나 급격하게 변하는지를 모델에게 알려주는 값입니다.
    - **설정 가이드**:
        - **낮은 값 (0.01 ~ 0.1)**: 안정적인 시장. 추세가 거의 변하지 않거나 완만하게 변할 때 사용합니다. (예: 쌀 가격)
        - **높은 값 (0.1 ~ 0.5)**: 역동적인 시장. 신제품 출시, 정책 변화 등으로 추세가 자주, 크게 꺾일 때 사용합니다. (예: 유행성 패션 아이템)
- **`계절성 강도 (seasonality_prior_scale)`**
    - **역할**: 1년 주기, 1주일 주기 등 반복되는 패턴(계절성)을 얼마나 강하게 믿을지를 결정합니다.
    - **설정 가이드**:
        - **낮은 값 (0.1 ~ 1.0)**: 계절성이 약하거나 불규칙할 때 사용합니다.
        - **높은 값 (1.0 ~ 10.0)**: 여름휴가, 연말 쇼핑 시즌처럼 매년 뚜렷한 패턴이 반복될 때 사용합니다.
- **`계절성 모드 (seasonality_mode)`**
    - **역할**: 계절성 패턴이 전체 추세에 어떻게 영향을 미치는지를 정의합니다.
    - **`additive (덧셈 모드)`**: 계절성의 변동폭이 추세와 상관없이 **일정할 때** 사용합니다. (예: 매년 여름 매출이 약 '1천만 원'씩 증가)
    - **`multiplicative (곱셈 모드)`**: 계절성의 변동폭이 추세에 따라 **함께 커지거나 작아질 때** 사용합니다. (예: 비즈니스가 성장함에 따라, 매년 여름 매출이 '10%'씩 증가)

#### **나. Prophet 예측 결과, 어떻게 해석해야 할까요?**
- **Prophet 예측 그래프**
    - **정확도 판단 기준**: "예측선(파란선)이 과거의 실제 데이터(검은 점)의 전반적인 흐름을 잘 따라가는가?"가 1차적인 기준입니다. 특히, 최근 데이터의 패턴을 잘 맞추고 있는지 확인하는 것이 중요합니다. '불확실성 구간(하늘색 영역)'이 너무 넓지 않고, 실제 값들이 대부분 그 안에 있다면 신뢰할 만한 예측입니다.
- **요인 분해 그래프**
    - **`trend`**: 데이터의 장기적인 방향성입니다. 이 선이 우상향하면 장기적으로 사업이 성장하고 있음을, 우하향하면 축소되고 있음을 의미합니다.
    - **`yearly` / `weekly`**: 해당 주기의 '순수한' 영향력을 보여줍니다. 예를 들어, `yearly` 그래프가 7월에 가장 높다면, 다른 모든 요인을 제외하고도 "7월"이라는 시점 자체가 예측값을 끌어올리는 효과가 있다는 뜻입니다.
    - **`extra_regressors` (외부 예측 변수)**: 해당 외부 변수가 예측값에 미친 **'추가적인 영향력'**을 보여줍니다. 예를 들어, 선물 가격(regressor) 그래프가 양수(+) 값을 보이면, 그 시점의 높은 선물 가격이 최종 예측값을 끌어올리는 역할을 했다는 의미입니다.

#### **다. 모델 진단: 내 예측 모델, 믿을만 한가요?**
- **ADF Test**
    - **정의**: 모델이 놓친 **"예측 오차(잔차)"에 여전히 유의미한 패턴이 남아있는지**를 검사하는 '패턴 탐지기'입니다.
    - **안정성 판단**: P-value가 **`0.05` 미만**이면 "오차에 패턴이 남아있을 확률이 5% 미만이다", 즉 **"오차는 거의 무작위적이므로 모델이 안정적이다"**라고 결론 내립니다. 반대로 0.05 이상이면, 아직 모델이 학습하지 못한 패턴이 남아있어 개선이 필요하다는 신호입니다.

#### **라. 최종 예측 (XGBoost): 더 깊은 인사이트 발견하기**
- **`R² Score` (설명력 / 결정계수)**
    - **의미**: 모델이 **"얼마나 미래를 잘 설명하는가?"**를 0~1 사이의 점수로 나타냅니다. 0.7 이라는 값은, 우리 모델이 미래 변동성의 70%를 설명할 수 있다는 뜻입니다.
    - **의미 있는 수준**: 일반적으로 **0.6 이상**이면 준수한 모델, **0.8 이상**이면 매우 좋은 모델로 평가합니다. **음수(-)가 나오면, 단순히 평균값으로 예측하는 것보다도 성능이 나쁘다는 최악의 신호입니다.**
- **`RMSE` (평균 오차)**
    - **의미**: **"그래서 예측이 평균적으로 얼마나 틀렸는가?"**를 실제 단위로 보여줍니다. 낮을수록 좋은 모델입니다.
"""

# ----------------------------
#  Helper Functions (Analysis & Interpretation)
# ----------------------------
@st.cache_data
def calculate_advanced_correlation(df, method='pearson'):
    df_numeric = df.select_dtypes(include=np.number).dropna(how='all', axis=1)
    cols = df_numeric.columns
    corr_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    pval_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    pvalues_list = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1_data, col2_data = df_numeric[cols[i]].dropna(), df_numeric[cols[j]].dropna()
            common_index = col1_data.index.intersection(col2_data.index)
            if len(common_index) < 3: continue
            col1_data, col2_data = col1_data.loc[common_index], col2_data.loc[common_index]
            corr, pval = (pearsonr if method == 'pearson' else spearmanr)(col1_data, col2_data)
            corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i] = corr
            pval_matrix.iloc[i, j] = pval_matrix.iloc[j, i] = pval
            pvalues_list.append(pval)
    if pvalues_list:
        _, pvals_corrected, _, _ = multipletests(pvalues_list, alpha=0.05, method='fdr_bh')
        k = 0
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pval_matrix.iloc[i, j] = pval_matrix.iloc[j, i] = pvals_corrected[k]; k += 1
    return corr_matrix, pval_matrix

@st.cache_data
def find_best_lagged_correlation(df, driver_vars, outcome_vars, max_lag=12):
    best_correlations = []
    for driver in driver_vars:
        for outcome in outcome_vars:
            if driver == outcome: continue
            max_corr_val, best_lag = 0, 0
            for lag in range(-max_lag, max_lag + 1):
                try:
                    corr = df[driver].shift(lag).corr(df[outcome])
                    if pd.notna(corr) and abs(corr) > abs(max_corr_val):
                        max_corr_val, best_lag = corr, lag
                except Exception: continue
            if best_lag != 0:
                best_correlations.append({'Driver (X)': driver, 'Outcome (Y)': outcome, 'Best Lag (Weeks)': best_lag, 'Correlation': max_corr_val})
    if not best_correlations: return pd.DataFrame()
    df_lags = pd.DataFrame(best_correlations)
    df_lags['Abs Correlation'] = df_lags['Correlation'].abs()
    return df_lags.sort_values('Abs Correlation', ascending=False).drop(columns=['Abs Correlation'])

def interpret_correlation(corr_matrix, pval_matrix, threshold=0.05):
    strong_corrs = []
    driver_keywords = ['news', 'naver', 'sentiment', '가격', 'futures']
    outcome_keywords = ['수입', 'volume']
    driver_cols = [c for c in corr_matrix.columns if any(kw in c.lower() for kw in driver_keywords)]
    outcome_cols = [c for c in corr_matrix.columns if any(kw in c.lower() for kw in outcome_keywords)]
    for driver in driver_cols:
        for outcome in outcome_cols:
            if driver == outcome or outcome not in corr_matrix.index or driver not in corr_matrix.columns: continue
            corr_val, pval = corr_matrix.loc[driver, outcome], pval_matrix.loc[driver, outcome]
            if pval < threshold and abs(corr_val) >= 0.4:
                direction = "양의" if corr_val > 0 else "음의"
                interpretation = f"**'{driver}'**와(과) **'{outcome}'** 사이에는 통계적으로 유의미한 **{direction} 상관관계**가 있습니다 (상관계수: {corr_val:.2f}, p-value: {pval:.3f})."
                strong_corrs.append({'text': interpretation, 'value': abs(corr_val)})
    if not strong_corrs: return "Driver(감성, 검색량 등)와 Outcome(수입량, 금액 등) 변수 그룹 간에 통계적으로 의미 있는 강한 관계는 발견되지 않았습니다."
    strong_corrs = sorted(strong_corrs, key=lambda x: x['value'], reverse=True)
    summary = "### 주요 발견 (Driver vs Outcome):\n\n" + "\n".join([f"- {corr['text']}" for corr in strong_corrs[:5]])
    summary += "\n\n* **양의 상관관계 (+):** Driver 변수가 증가할 때 Outcome 변수도 함께 증가하는 경향을 보입니다."
    summary += "\n* **음의 상관관계 (-):** Driver 변수가 증가할 때 Outcome 변수는 오히려 감소하는 경향을 보입니다."
    return summary

# ----------------------------
#  Helpers: Data Fetching & Processing
# ----------------------------
@st.cache_resource
def get_bq_connection():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        return bigquery.Client(credentials=creds, project=creds.project_id)
    except Exception as e:
        st.error(f"Google BigQuery 연결 실패: {e}"); return None

def upload_df_to_bq(client, df, table_id):
    try:
        pandas_gbq.to_gbq(df, f"{BQ_DATASET}.{table_id}", project_id=client.project, if_exists="append", credentials=client._credentials)
        return True, None
    except Exception as e: return False, str(e)

@st.cache_resource
def load_models(_model_ids, token):
    models = {}
    for key, mid in _model_ids.items():
        try:
            models[key] = pipeline("sentiment-analysis", model=mid, tokenizer=mid, token=token)
        except Exception: models[key] = None
    return models

@st.cache_data(ttl=3600)
def get_news_with_multi_model_analysis(_bq_client, _models_dict, keyword):
    all_news = []
    rss_url = f"https://news.google.com/rss/search?q={quote(keyword)}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)
    for entry in feed.entries[:50]:
        title = entry.get('title', '').strip()
        if not title: continue
        pub_date = pd.to_datetime(entry.get('published')).date() if 'published' in entry else datetime.utcnow().date()
        all_news.append({"날짜": pub_date, "Title": title, "ModelInput": title})
    if not all_news: return pd.DataFrame()
    df_new = pd.DataFrame(all_news).drop_duplicates(subset=["ModelInput"])
    with st.spinner(f"'{keyword}' 뉴스 감성 분석 중..."):
        def _label_score_to_signed(pred):
            if not pred: return 0.0
            lbl, score = str(pred.get("label", "")).lower(), float(pred.get("score", 0.0))
            if "star" in lbl:
                try: n = int(lbl.split()[0]); return (n - 3) / 2.0
                except (ValueError, IndexError): return 0.0
            if any(x in lbl for x in ["neg", "negative", "부정"]): return -score
            if any(x in lbl for x in ["pos", "positive", "긍정"]): return score
            return 0.0
        preds = {key: (model(df_new['ModelInput'].tolist(), truncation=True, max_length=512) if model else [None]*len(df_new)) for key, model in _models_dict.items()}
        multi = []
        for i in range(len(df_new)):
            multi.append({
                "News_FinBERT": _label_score_to_signed(preds.get("finbert", [])[i]),
                "News_Elite": _label_score_to_signed(preds.get("elite", [])[i]),
                "News_Product": _label_score_to_signed(preds.get("product", [])[i])
            })
    return pd.concat([df_new.reset_index(drop=True), pd.DataFrame(multi)], axis=1).drop(columns=['ModelInput'])

@st.cache_data(ttl=3600)
def get_categories_from_bq(_client):
    try:
        query = f"SELECT DISTINCT Category FROM `{_client.project}.{BQ_DATASET}.{BQ_TABLE_TRADE}` ORDER BY Category"
        return sorted(_client.query(query).to_dataframe()['Category'].astype(str).unique())
    except Exception: return []

def get_trade_data_from_bq(client, categories):
    if not categories: return pd.DataFrame()
    try:
        sql = f"SELECT * FROM `{client.project}.{BQ_DATASET}.{BQ_TABLE_TRADE}` WHERE Category IN UNNEST(@categories)"
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("categories", "STRING", categories)])
        df = client.query(sql, job_config=job_config).to_dataframe()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception as e: st.error(f"BigQuery TDS 데이터 로드 오류: {e}"); return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_yfinance_data(ticker_name, ticker_symbol, start_date, end_date):
    data_frames = []
    current_start = start_date
    while current_start < end_date:
        current_end = current_start + timedelta(days=365 * 2)
        if current_end > end_date: current_end = end_date
        try:
            data = yf.download(ticker_symbol, start=current_start, end=current_end, progress=False)
            if not data.empty: data_frames.append(data)
        except Exception as e:
            st.sidebar.error(f"Yahoo Finance 데이터 일부 로드 중 오류: {e}")
        current_start = current_end + timedelta(days=1)
    if not data_frames:
        st.sidebar.warning(f"'{ticker_name}'에 대한 선물 데이터를 가져올 수 없습니다. 기간이나 티커를 확인해주세요.")
        return pd.DataFrame()
    full_data = pd.concat(data_frames)
    df = full_data[['Close']].rename(columns={'Close': f'Futures_{ticker_name}'})
    df.index.name = '날짜'
    return df

@st.cache_data(ttl=3600)
def fetch_kamis_data(item_info, start_date, end_date, kamis_keys):
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    url = (f"http://www.kamis.or.kr/service/price/xml.do?action=periodWholesaleProductList"
           f"&p_product_cls_code=01&p_startday={start_str}&p_endday={end_str}"
           f"&p_item_category_code={item_info['cat_code']}&p_item_code={item_info['item_code']}&p_kind_code={item_info['kind_code']}"
           f"&p_product_rank_code={item_info['rank_code']}&p_convert_kg_yn=Y"
           f"&p_cert_key={kamis_keys['key']}&p_cert_id={kamis_keys['id']}&p_returntype=json")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and "data" in response.json():
            price_data = response.json().get("data", {}).get("item", [])
            if not price_data: return pd.DataFrame()
            df_new = pd.DataFrame(price_data)[['regday', 'price']].rename(columns={'regday': '날짜', 'price': '도매가격_원'})
            df_new['날짜'] = pd.to_datetime(df_new['날짜'].str.replace('/', '-'))
            df_new['도매가격_원'] = pd.to_numeric(df_new['도매가격_원'].str.replace(',', ''))
            return df_new
    except Exception: return pd.DataFrame()

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
        st.sidebar.error(f"Naver API 오류 발생: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_naver_trends_data(_client, keywords, start_date, end_date, naver_keys):
    if not naver_keys.get('id') or not naver_keys.get('secret'): return pd.DataFrame()
    all_data = []
    for keyword in keywords:
        body = json.dumps({
            "startDate": start_date.strftime('%Y-%m-%d'), "endDate": end_date.strftime('%Y-%m-%d'),
            "timeUnit": "date", "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
        })
        response = call_naver_api("https://openapi.naver.com/v1/datalab/search", body, naver_keys)
        if response and response.get('results'):
            data = response['results'][0]['data']
            if data:
                df_keyword = pd.DataFrame(data).rename(columns={'period': '날짜', 'ratio': f'Naver_{keyword}'})
                df_keyword['날짜'] = pd.to_datetime(df_keyword['날짜'])
                all_data.append(df_keyword)
    if not all_data:
        st.sidebar.warning(f"Naver Trends에서 '{','.join(keywords)}' 데이터를 찾을 수 없습니다.")
        return pd.DataFrame()
    return reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_data) if len(all_data) > 1 else all_data[0]

@st.cache_data
def find_best_prophet_params(_df, _regressors):
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []
    initial_days = str(int(len(_df) * 0.5)) + ' days'
    period_days = str(int(len(_df) * 0.2)) + ' days'
    horizon_days = str(int(len(_df) * 0.2)) + ' days'
    if int(len(_df) * 0.2) < 30 : return all_params[1]
    for params in all_params:
        try:
            m = Prophet(**params)
            for reg in _regressors: m.add_regressor(reg)
            m.fit(_df)
            df_cv = cross_validation(m, initial=initial_days, period=period_days, horizon=horizon_days, parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        except Exception: rmses.append(float('inf'))
    best_params = all_params[np.argmin(rmses)]
    return best_params

# ----------------------------
#  App Main Logic
# ----------------------------
bq_client = get_bq_connection()
if bq_client is None: st.stop()
hf_token = st.secrets.get("huggingface", {}).get("token")
models = load_models(DEFAULT_MODEL_IDS, hf_token)
if 'final_df' not in st.session_state: st.session_state.final_df = pd.DataFrame()
if 'raw_news_df' not in st.session_state: st.session_state.raw_news_df = pd.DataFrame()
if 'best_params' not in st.session_state: st.session_state.best_params = {}

st.sidebar.header("⚙️ 분석 설정")
categories = get_categories_from_bq(bq_client)
selected_categories = st.sidebar.multiselect("분석할 품목 선택", categories, default=categories[:1] if categories else [])
start_date = pd.to_datetime(st.sidebar.date_input('시작일', datetime(2022, 1, 1)))
end_date = pd.to_datetime(st.sidebar.date_input('종료일', datetime.now()))
news_keyword_input = st.sidebar.text_input("뉴스 분석 키워드", selected_categories[0] if selected_categories else "")
naver_keywords_input = st.sidebar.text_input("네이버 트렌드 키워드 (쉼표 구분)", selected_categories[0] if selected_categories else "")
with st.sidebar.expander("🔑 API 키 입력 (선택)"):
    naver_client_id = st.text_input("Naver API Client ID", type="password")
    naver_client_secret = st.text_input("Naver API Client Secret", type="password")
    kamis_api_key = st.text_input("KAMIS API Key", type="password")
    kamis_api_id = st.text_input("KAMIS API ID", type="password")
st.sidebar.subheader("🌍 외부 가격 데이터 소스")
price_source = st.sidebar.radio("가격 소스 선택", ["KAMIS 국내 도매가격", "Yahoo Finance 선물 가격"])
if price_source == "KAMIS 국내 도매가격":
    kamis_item_name = st.sidebar.selectbox("품목 선택", list(KAMIS_FULL_DATA.keys()))
    kamis_kind_name = st.sidebar.selectbox("품종 선택", list(KAMIS_FULL_DATA[kamis_item_name]['kinds'].keys())) if kamis_item_name else ""
else: selected_commodity = st.sidebar.selectbox("선물 품목 선택", list(COMMODITY_TICKERS.keys()))

if st.sidebar.button("🚀 모든 데이터 통합 및 분석 실행"):
    with st.spinner("데이터 통합 및 분석 중..."):
        trade_df = get_trade_data_from_bq(bq_client, selected_categories)
        news_df = get_news_with_multi_model_analysis(bq_client, models, news_keyword_input)
        st.session_state.raw_news_df = news_df
        naver_keys = {'id': naver_client_id, 'secret': naver_client_secret}
        naver_keywords = [k.strip() for k in naver_keywords_input.split(',') if k.strip()]
        naver_df = fetch_naver_trends_data(bq_client, naver_keywords, start_date, end_date, naver_keys)
        external_price_df = pd.DataFrame()
        if price_source == "KAMIS 국내 도매가격":
            if kamis_item_name and kamis_kind_name and kamis_api_key and kamis_api_id:
                kamis_keys = {'key': kamis_api_key, 'id': kamis_api_id}
                item_info = KAMIS_FULL_DATA[kamis_item_name].copy()
                item_info['kind_code'] = item_info['kinds'][kamis_kind_name]; item_info['rank_code'] = '01'
                external_price_df = fetch_kamis_data(item_info, start_date, end_date, kamis_keys)
        else:
            if selected_commodity:
                ticker = COMMODITY_TICKERS[selected_commodity]
                external_price_df = fetch_yfinance_data(selected_commodity, ticker, start_date, end_date)
        trade_weekly = trade_df.set_index('Date').resample('W-Mon').agg(수입액_USD=('Value', 'sum'), 수입량_KG=('Volume', 'sum')).copy()
        trade_weekly['수입단가_USD_KG'] = trade_weekly['수입액_USD'] / trade_weekly['수입량_KG']
        dfs_to_merge = [trade_weekly]
        if not news_df.empty:
            news_df['날짜'] = pd.to_datetime(news_df['날짜'])
            dfs_to_merge.append(news_df.drop(columns=['Title']).set_index('날짜').resample('W-Mon').mean())
        if not naver_df.empty:
            naver_df['날짜'] = pd.to_datetime(naver_df['날짜'])
            dfs_to_merge.append(naver_df.set_index('날짜').resample('W-Mon').mean())
        if not external_price_df.empty:
            if '날짜' in external_price_df.columns:
                external_price_df['날짜'] = pd.to_datetime(external_price_df['날짜'])
                external_price_df = external_price_df.set_index('날짜')
            dfs_to_merge.append(external_price_df.resample('W-Mon').mean())
        final_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs_to_merge)
        final_df = final_df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
        st.session_state.final_df = final_df.replace([np.inf, -np.inf], np.nan).dropna()
        st.session_state.best_params = {}
        st.success("데이터 통합 완료!")

if not st.session_state.final_df.empty:
    final_df = st.session_state.final_df
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 상관관계 분석", "📈 시계열 예측", "📄 통합 데이터", "📰 수집 뉴스 원본", "📘 대시보드 사용법"])
    with tab1:
        st.header("상관관계 분석")
        col1, col2 = st.columns(2)
        corr_method = col1.selectbox("상관관계 분석 방법", ('pearson', 'spearman'), key="corr_method")
        pval_threshold = col2.slider("유의수준 (P-value) 필터", 0.0, 1.0, 0.05, key="pval_slider")
        corr_matrix, pval_matrix = calculate_advanced_correlation(final_df, method=corr_method)
        st.subheader(f"'{corr_method.capitalize()}' 상관관계 히트맵")
        fig_heatmap = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu_r', zmin=-1, zmax=1, text=corr_matrix.round(2).astype(str), texttemplate="%{text}"))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        with st.expander("🔍 히트맵 결과 해석하기"):
            st.markdown(interpret_correlation(corr_matrix, pval_matrix, threshold=pval_threshold))
        st.markdown("---")
        st.subheader("시차 교차상관 분석")
        driver_cols = st.multiselect("선행 변수 (Driver)", final_df.columns, default=[c for c in final_df if any(kw in c.lower() for kw in ['news', 'naver', '가격', 'futures'])])
        outcome_cols = st.multiselect("후행 변수 (Outcome)", final_df.columns, default=[c for c in final_df if '수입' in c])
        if driver_cols and outcome_cols:
            lag_df = find_best_lagged_correlation(final_df, driver_cols, outcome_cols)
            st.dataframe(lag_df.head(10))
            if not lag_df.empty:
                st.info(f"가장 강한 시차 관계: **{lag_df.iloc[0]['Driver (X)']}**의 변화는 **{lag_df.iloc[0]['Best Lag (Weeks)']}주 후** **{lag_df.iloc[0]['Outcome (Y)']}**에 영향을 미치는 경향이 있습니다 (상관계수: {lag_df.iloc[0]['Correlation']:.3f}).")
        st.markdown("---")
        st.subheader("산점도 행렬 (Scaled for Visualization)")
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(final_df), columns=final_df.columns, index=final_df.index)
        dims = st.multiselect("산점도 표시 변수", scaled_df.columns, default=list(scaled_df.columns[:8]))
        if dims:
            st.plotly_chart(px.scatter_matrix(scaled_df[dims]), use_container_width=True)
    with tab2:
        st.header("시계열 예측 (Prophet & XGBoost)")
        prophet_df = final_df.reset_index().rename(columns={final_df.index.name if final_df.index.name else 'index': 'ds'})
        col1, col2 = st.columns(2)
        forecast_col = col1.selectbox("예측 대상 변수 (y)", final_df.columns, key="forecast_col")
        forecast_periods = col2.number_input("예측 기간 (주)", 4, 52, 12, key="forecast_periods")
        prophet_df = prophet_df.rename(columns={forecast_col: 'y'})
        regressors = [c for c in prophet_df.columns if c not in ['ds', 'y']]
        selected_regressors = st.multiselect("외부 예측 변수 (Regressors)", regressors, default=regressors, key="regressors")
        st.subheader("Prophet 모델 파라미터 튜닝")
        if st.button("🤖 최적 파라미터 자동 탐색"):
            with st.spinner("교차 검증을 통해 최적의 파라미터를 탐색 중입니다... (데이터 양에 따라 1~2분 소요될 수 있습니다)"):
                best_params = find_best_prophet_params(prophet_df[['ds', 'y'] + selected_regressors], selected_regressors)
                st.session_state.best_params = best_params
                st.success("최적 파라미터 탐색 완료!")
        bp = st.session_state.best_params
        p_col1, p_col2, p_col3 = st.columns(3)
        changepoint_prior_scale = p_col1.slider("Trend 유연성", 0.01, 0.5, bp.get('changepoint_prior_scale', 0.05), key="cps")
        seasonality_prior_scale = p_col2.slider("계절성 강도", 0.01, 10.0, bp.get('seasonality_prior_scale', 1.0), key="sps")
        seasonality_mode_options = ['additive', 'multiplicative']
        seasonality_mode_index = seasonality_mode_options.index(bp.get('seasonality_mode', 'additive'))
        seasonality_mode = p_col3.selectbox("계절성 모드", seasonality_mode_options, index=seasonality_mode_index, key="sm")
        if st.button("🚀 예측 실행", key="run_forecast"):
            m = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale, seasonality_mode=seasonality_mode)
            for reg in selected_regressors: m.add_regressor(reg)
            m.fit(prophet_df[['ds', 'y'] + selected_regressors])
            future = m.make_future_dataframe(periods=forecast_periods, freq='W')
            future_regressors = prophet_df[['ds'] + selected_regressors].set_index('ds'); last_values = future_regressors.iloc[-1]; future_regressors = future_regressors.reindex(future['ds']).fillna(method='ffill').fillna(last_values); future = pd.concat([future.set_index('ds'), future_regressors], axis=1).reset_index()
            forecast = m.predict(future)
            st.subheader("Prophet 예측 결과"); st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
            st.subheader("Prophet 요인 분해"); st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)
            st.markdown("---"); st.subheader("모델 진단: 잔차 분석")
            df_pred = forecast.set_index('ds')[['yhat']].join(prophet_df.set_index('ds')[['y']]).dropna(); residuals = df_pred['y'] - df_pred['yhat']
            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                st.markdown("**잔차 정상성 검정 (ADF Test)**"); adf_result = adfuller(residuals); st.write(f"p-value: {adf_result[1]:.4f}")
            with diag_col2:
                st.markdown("**잔차 분포**"); st.plotly_chart(ff.create_distplot([residuals], ['residuals'], bin_size=.2, show_rug=False), use_container_width=True)
            st.markdown("---"); st.subheader("고급 예측: XGBoost Meta-Forecasting")
            ml_df = forecast[['ds', 'trend']].set_index('ds'); available_seasonal_components = []
            if 'yearly' in forecast.columns: ml_df = ml_df.join(forecast[['ds', 'yearly']].set_index('ds')); available_seasonal_components.append('yearly')
            if 'weekly' in forecast.columns: ml_df = ml_df.join(forecast[['ds', 'weekly']].set_index('ds')); available_seasonal_components.append('weekly')
            ml_df = ml_df.join(prophet_df.set_index('ds')).dropna()
            X = ml_df[['trend'] + available_seasonal_components + selected_regressors]; y = ml_df['y']
            train_size = int(len(X) * 0.85); X_train, X_test, y_train, y_test = X.iloc[:train_size], X.iloc[train_size:], y.iloc[:train_size], y.iloc[train_size:]
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, early_stopping_rounds=50); xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred_xgb = xgb_model.predict(X_test); r2, rmse = r2_score(y_test, y_pred_xgb), np.sqrt(mean_squared_error(y_test, y_pred_xgb))
            st.metric("XGBoost Test R² Score", f"{r2:.3f}"); st.metric("XGBoost Test RMSE", f"{rmse:.3f}")
            fig_xgb = go.Figure(); fig_xgb.add_trace(go.Scatter(x=y_train.index, y=y_train, name='Train')); fig_xgb.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Test (Actual)')); fig_xgb.add_trace(go.Scatter(x=y_test.index, y=y_pred_xgb, name='XGBoost Prediction'))
            st.plotly_chart(fig_xgb, use_container_width=True)
            feature_imp = pd.DataFrame(sorted(zip(xgb_model.feature_importances_, X.columns)), columns=['Value','Feature'])
            st.plotly_chart(px.bar(feature_imp, x="Value", y="Feature", orientation='h', title="Feature Importance"), use_container_width=True)
            with st.expander("🔍 XGBoost 종합 결과 해석"): st.markdown(guide_content.split("`R² Score`")[1])
    with tab3:
        st.header("통합 데이터 (주별)")
        st.dataframe(final_df)
        st.download_button(label="CSV로 다운로드", data=final_df.to_csv(index=False).encode('utf-8-sig'), file_name="integrated_weekly_data.csv", mime='text/csv',)
    with tab4:
        st.header("📰 수집 뉴스 원본")
        if not st.session_state.raw_news_df.empty:
            st.dataframe(st.session_state.raw_news_df.sort_values(by='날짜', ascending=False))
        else:
            st.info("사이드바에서 분석을 실행하면 수집된 뉴스 기사 제목과 감성 점수를 여기서 확인할 수 있습니다.")
    with tab5:
        st.header("📘 대시보드 사용법 가이드")
        st.markdown(guide_content, unsafe_allow_html=True)
else:
    st.info("👈 사이드바에서 분석할 데이터를 선택하고 '모든 데이터 통합 및 분석 실행' 버튼을 눌러주세요.")

