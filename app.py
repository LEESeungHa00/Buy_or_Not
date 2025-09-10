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
from kamis_data import KAMIS_FULL_DATA

# Advanced Analysis Libraries
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

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
BQ_TABLE_NAVER = "naver_trends_cache"
BQ_TABLE_NEWS = "news_sentiment_finbert"
BQ_TABLE_TRADE = "tds_data"

# ----------------------------
#  Helper Functions (Analysis & Interpretation)
# ----------------------------
@st.cache_data
def calculate_advanced_correlation(df, method='pearson', p_adjust_method='fdr_bh'):
    """Calculate correlation matrix and corresponding p-values."""
    df_numeric = df.select_dtypes(include=np.number).dropna(how='all', axis=1)
    cols = df_numeric.columns
    corr_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    pval_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    
    pvalues_list = []
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1_data = df_numeric[cols[i]].dropna()
            col2_data = df_numeric[cols[j]].dropna()
            
            common_index = col1_data.index.intersection(col2_data.index)
            if len(common_index) < 3: continue

            col1_data = col1_data.loc[common_index]
            col2_data = col2_data.loc[common_index]

            if method == 'pearson':
                corr, pval = pearsonr(col1_data, col2_data)
            elif method == 'spearman':
                corr, pval = spearmanr(col1_data, col2_data)
            else: # kendall
                corr, pval = kendalltau(col1_data, col2_data)
                
            corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i] = corr
            pval_matrix.iloc[i, j] = pval_matrix.iloc[j, i] = pval
            pvalues_list.append(pval)

    # Adjust p-values for multiple comparisons
    if pvalues_list:
        _, pvals_corrected, _, _ = multipletests(pvalues_list, alpha=0.05, method=p_adjust_method)
        
        corrected_pval_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        k = 0
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corrected_pval_matrix.iloc[i, j] = corrected_pval_matrix.iloc[j, i] = pvals_corrected[k]
                k += 1
    else:
        corrected_pval_matrix = pval_matrix

    return corr_matrix, corrected_pval_matrix


@st.cache_data
def find_best_lagged_correlation(df, driver_vars, outcome_vars, max_lag=12):
    """Find the best time lag for correlation between driver and outcome variables."""
    best_correlations = []
    for driver in driver_vars:
        for outcome in outcome_vars:
            if driver == outcome: continue
            
            max_corr_val = 0
            best_lag = 0
            
            for lag in range(-max_lag, max_lag + 1):
                try:
                    shifted_driver = df[driver].shift(lag)
                    corr = shifted_driver.corr(df[outcome])
                except Exception:
                    corr = np.nan

                if pd.notna(corr) and abs(corr) > abs(max_corr_val):
                    max_corr_val = corr
                    best_lag = lag
            
            if best_lag != 0:
                best_correlations.append({
                    'Driver (X)': driver,
                    'Outcome (Y)': outcome,
                    'Best Lag (Weeks)': best_lag,
                    'Correlation': max_corr_val
                })
    
    if not best_correlations:
        return pd.DataFrame()
        
    df_lags = pd.DataFrame(best_correlations)
    df_lags['Abs Correlation'] = df_lags['Correlation'].abs()
    return df_lags.sort_values('Abs Correlation', ascending=False).drop(columns=['Abs Correlation'])

def interpret_correlation_groups(corr_matrix, pval_matrix, threshold=0.05):
    """Generates a human-readable interpretation of correlations between driver and outcome groups."""
    driver_keywords = ['Naver', 'News_', 'Sentiment', '도매']
    outcome_keywords = ['수입']

    all_cols = corr_matrix.columns
    driver_cols = [col for col in all_cols if any(key in col for key in driver_keywords)]
    outcome_cols = [col for col in all_cols if any(key in col for key in outcome_keywords)]
    
    driver_cols = [col for col in driver_cols if col not in outcome_cols]

    if not driver_cols or not outcome_cols:
        return "자동 해석을 위한 'Driver(원인)' 또는 'Outcome(결과)' 그룹의 변수가 부족합니다. (Driver: 'Naver', 'News', '도매' 등 / Outcome: '수입' 등)"

    strong_corrs = []
    for driver in driver_cols:
        for outcome in outcome_cols:
            if driver not in corr_matrix.index or outcome not in corr_matrix.columns:
                continue

            corr_val = corr_matrix.loc[driver, outcome]
            pval = pval_matrix.loc[driver, outcome]
            
            if pval < threshold and abs(corr_val) >= 0.4: # Slightly lower threshold for discovery
                direction = "강한 양의" if corr_val > 0 else "강한 음의"
                interpretation = (
                    f"**'{driver}'** (원인)와(과) **'{outcome}'** (결과) 사이에는 "
                    f"통계적으로 유의미한 **{direction} 상관관계**가 있습니다 (상관계수: {corr_val:.2f}, p-value: {pval:.3f})."
                )
                strong_corrs.append({'text': interpretation, 'value': abs(corr_val)})
    
    if not strong_corrs:
        return "현재 설정된 유의수준에서 Driver(원인) 변수와 Outcome(결과) 변수 간의 의미 있는 관계는 발견되지 않았습니다. 변수 선택이나 기간을 변경하여 다시 분석해 보세요."
        
    strong_corrs = sorted(strong_corrs, key=lambda x: x['value'], reverse=True)
    
    summary = "### 주요 발견 (Driver vs. Outcome):\n\n" + "\n".join([f"- {corr['text']}" for corr in strong_corrs[:5]])
    summary += "\n\n* **양의 상관관계 (+):** 원인 변수가 증가할 때 결과 변수도 함께 증가하는 경향을 보입니다."
    summary += "\n* **음의 상관관계 (-):** 원인 변수가 증가할 때 결과 변수는 오히려 감소하는 경향을 보입니다."
    return summary

def interpret_adf_test(adf_result):
    """Generates an interpretation of the ADF test result."""
    p_value = adf_result[1]
    if p_value < 0.05:
        return (
            f"**결론: 모델이 안정적일 가능성이 높습니다.** (p-value: {p_value:.3f})\n\n"
            "ADF 테스트 결과, 예측 오차(잔차)들이 특정 패턴 없이 무작위적인 '백색소음'에 가깝다는 것을 의미합니다. "
            "이는 Prophet 모델이 데이터의 주요 추세와 계절성 패턴을 성공적으로 학습했다는 긍정적인 신호입니다."
        )
    else:
        return (
            f"**결론: 모델 개선의 여지가 있습니다.** (p-value: {p_value:.3f})\n\n"
            "예측 오차(잔차)에 아직 설명되지 않은 특정 패턴이 남아있을 수 있음을 시사합니다. "
            "이는 모델이 데이터의 모든 정보를 완전히 학습하지 못했을 가능성을 의미합니다. "
            "예측력을 높이기 위해 다른 외부 변수(Regressor)를 추가하거나 Prophet 파라미터를 조정하는 것을 고려해볼 수 있습니다."
        )

def interpret_xgboost_results(r2, rmse, feature_imp_df, y_mean):
    """Generates an interpretation of XGBoost model results."""
    r2_perc = r2 * 100
    rmse_perc = (rmse / y_mean) * 100 if y_mean != 0 else 0

    top_features = feature_imp_df.sort_values('Value', ascending=False).head(3)['Feature'].tolist()

    interpretation = f"""
    ### XGBoost 모델 성능 요약:

    - **R² Score (설명력): {r2:.3f}**
      - **의미:** 우리 모델이 예측 대상 변수의 전체 변동성 중 **약 {r2_perc:.1f}%**를 성공적으로 설명하고 있습니다.
      - **판단:** 1에 가까울수록 좋습니다. 이 수치가 높다는 것은 모델이 데이터의 패턴을 잘 포착하고 있다는 의미입니다.

    - **RMSE (평균 예측 오차): {rmse:.3f}**
      - **의미:** 모델의 예측값은 실제값과 평균적으로 **약 {rmse:.3f}** 만큼의 오차를 보입니다. (이는 예측 대상 평균값의 약 {rmse_perc:.1f}% 수준입니다.)
      - **판단:** 0에 가까울수록 좋습니다. 이 수치가 낮다는 것은 예측이 더 정밀하다는 것을 의미합니다.

    ### 예측에 가장 큰 영향을 미친 변수 TOP 3:
    1. **{top_features[0]}**
    2. **{top_features[1]}**
    3. **{top_features[2]}**

    **종합 의견:**
    이 변수들이 미래 값을 예측하는 데 가장 중요한 역할을 했습니다. 비즈니스 전략 수립 시 이 핵심 지표들의 변화를 주의 깊게 모니터링하는 것이 중요합니다.
    """
    return interpretation

# ----------------------------
#  Helpers: BigQuery + Naver + KAMIS
# ----------------------------
@st.cache_resource
def get_bq_connection():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        client = bigquery.Client(credentials=creds, project=creds.project_id)
        return client
    except Exception as e:
        st.error(f"Google BigQuery 연결 실패: secrets.toml을 확인하세요. 오류: {e}")
        return None

def upload_df_to_bq(client, df, table_id):
    """Uploads a DataFrame to a specified BigQuery table."""
    try:
        pandas_gbq.to_gbq(
            df,
            f"{BQ_DATASET}.{table_id}",
            project_id=client.project,
            if_exists="append",
            credentials=client._credentials
        )
        return True, None
    except Exception as e:
        return False, str(e)

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
        st.error(f"Naver API 오류 발생: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_kamis_data(_client, item_info, start_date, end_date, kamis_keys):
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    url = (f"http://www.kamis.or.kr/service/price/xml.do?action=periodWholesaleProductList"
           f"&p_product_cls_code=01&p_startday={start_str}&p_endday={end_str}"
           f"&p_item_category_code={item_info['cat_code']}&p_item_code={item_info['item_code']}&p_kind_code={item_info['kind_code']}"
           f"&p_product_rank_code={item_info['rank_code']}&p_convert_kg_yn=Y"
           f"&p_cert_key={kamis_keys['key']}&p_cert_id={kamis_keys['id']}&p_returntype=json")
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 KAMISClient/1.0"})
        if response.status_code == 200 and "data" in response.json() and "item" in response.json()["data"]:
            price_data = response.json()["data"]["item"]
            if not price_data:
                return pd.DataFrame()
            df_new = pd.DataFrame(price_data)[['regday', 'price']].rename(columns={'regday': '날짜', 'price': '도매가격_원'})
            def format_kamis_date(date_str):
                processed_str = str(date_str).replace('/', '-')
                if processed_str.count('-') == 1:
                    return f"{start_date.year}-{processed_str}"
                return processed_str
            df_new['날짜'] = pd.to_datetime(df_new['날짜'].apply(format_kamis_date), errors='coerce')
            df_new['도매가격_원'] = pd.to_numeric(df_new['도매가격_원'].astype(str).str.replace(',', ''), errors='coerce')
            return df_new
    except Exception as e:
        st.sidebar.error(f"KAMIS API 호출 중 오류: {e}")
    return pd.DataFrame()
# ----------------------------
#  HuggingFace: 로그인 + 안전한 모델 로드
# ----------------------------
hf_token = st.secrets.get("huggingface", {}).get("token")
if hf_token:
    try:
        hf_login(token=hf_token)
    except Exception as e:
        st.sidebar.warning(f"HuggingFace 로그인 실패: {e}")

DEFAULT_MODEL_IDS = {
    "finbert": "snunlp/KR-FinBERT-SC",
    "elite": "nlptown/bert-base-multilingual-uncased-sentiment",
    "product": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
}
@st.cache_resource
def load_models(model_ids, token):
    models = {}
    for key, mid in model_ids.items():
        try:
            models[key] = pipeline("sentiment-analysis", model=mid, tokenizer=mid, token=token)
        except Exception:
            models[key] = None
    return models
models = load_models(DEFAULT_MODEL_IDS, hf_token)

# ----------------------------
#  감성분석 점수 변환 함수
# ----------------------------
def _label_score_to_signed(pred):
    if not pred: return 0.0, "neutral"
    lbl, score = str(pred.get("label", "")).lower(), float(pred.get("score", 0.0))
    if "star" in lbl:
        try:
            n = int(lbl.split()[0])
            return (n - 3) / 2.0, "positive" if n > 3 else "negative" if n < 3 else "neutral"
        except (ValueError, IndexError):
            return 0.0, "neutral"
    if any(x in lbl for x in ["neg", "negative", "부정"]): return -score, "negative"
    if any(x in lbl for x in ["pos", "positive", "긍정"]): return score, "positive"
    return 0.0, "neutral"

def analyze_sentiment_multi(texts, _models_dict):
    results = []
    if not texts: return results
    
    preds = {key: (model(texts, truncation=True, max_length=512) if model else [None]*len(texts)) for key, model in _models_dict.items()}
        
    for i, _ in enumerate(texts):
        fin_s, fin_l = _label_score_to_signed(preds.get("finbert", [])[i] if preds.get("finbert") else None)
        el_s, el_l = _label_score_to_signed(preds.get("elite", [])[i] if preds.get("elite") else None)
        pr_s, pr_l = _label_score_to_signed(preds.get("product", [])[i] if preds.get("product") else None)
        results.append({
            "FinBERT_Sentiment": fin_s, "FinBERT_Label": fin_l,
            "Elite_Sentiment": el_s, "Elite_Label": el_l,
            "Product_Sentiment": pr_s, "Product_Label": pr_l,
            "Sentiment": fin_s, "Label": fin_l
        })
    return results
# ----------------------------
#  뉴스 수집/분석
# ----------------------------
def _fetch_article_text(url):
    if not _HAS_NEWSPAPER or not url: return ""
    try:
        cfg = NewsConfig()
        cfg.browser_user_agent = "Mozilla/5.0 (compatible; NewsBot/1.0)"
        cfg.request_timeout = 10
        art = Article(url, language='ko', config=cfg)
        art.download()
        art.parse()
        return art.text or ""
    except Exception:
        return ""

@st.cache_data(ttl=3600)
def get_news_with_multi_model_analysis(_bq_client, _models_dict, keyword, days_limit=7):
    project_id = _bq_client.project
    full_table_id = f"{project_id}.{BQ_DATASET}.{BQ_TABLE_NEWS}"
    try:
        time_limit = datetime.now(timezone.utc) - timedelta(days=days_limit)
        query = "SELECT * FROM `{}` WHERE Keyword = @keyword AND InsertedAt >= @time_limit ORDER BY 날짜 DESC".format(full_table_id)
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("keyword", "STRING", keyword),
            bigquery.ScalarQueryParameter("time_limit", "TIMESTAMP", time_limit)
        ])
        df_cache = _bq_client.query(query, job_config=job_config).to_dataframe()
    except Exception:
        df_cache = pd.DataFrame()
    if not df_cache.empty:
        return df_cache
    all_news = []
    rss_url = f"https://news.google.com/rss/search?q={quote(keyword)}&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)
    for entry in feed.entries[:60]:
        title = entry.get('title', '').strip()
        link = entry.get('link', '').strip()
        if not title: continue
        pub_date = pd.to_datetime(entry.get('published')).date() if 'published' in entry else datetime.utcnow().date()
        body = _fetch_article_text(link)
        text_for_model = (title + ". " + body).strip() if body else title
        all_news.append({"날짜": pub_date, "Title": title, "RawUrl": link, "ModelInput": text_for_model})
    if not all_news:
        st.error(f"'{keyword}'에 대한 뉴스를 찾지 못했습니다.")
        return pd.DataFrame()
    df_new = pd.DataFrame(all_news).drop_duplicates(subset=["Title"])
    with st.spinner(f"다중 모델로 '{keyword}' 뉴스 감성 분석 중 ({len(df_new)}건)..."):
        multi = analyze_sentiment_multi(df_new['ModelInput'].tolist(), _models_dict)
    multi_df = pd.DataFrame(multi)
    df_new = pd.concat([df_new.reset_index(drop=True), multi_df], axis=1)
    df_new['Keyword'] = keyword
    df_new['InsertedAt'] = datetime.now(timezone.utc)
    try:
        df_to_gbq = df_new.drop(columns=['ModelInput'])
        upload_df_to_bq(_bq_client, df_to_gbq, BQ_TABLE_NEWS)
    except Exception as e:
        st.sidebar.warning(f"BigQuery 뉴스 저장 실패: {e}")
    return df_new.drop(columns=['ModelInput'])

# ----------------------------
#  BigQuery 데이터 로드 함수들
# ----------------------------
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
        for col in [c for c in df.columns if 'price' in c.lower() or 'value' in c.lower() or 'volume' in c.lower()]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"BigQuery TDS 데이터 로드 오류: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_naver_trends_data(_client, keywords, start_date, end_date, naver_keys):
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
        last_cached_date = df_cache['날짜'].max().date()
        if start_date.date() > last_cached_date:
            fetch_start_date = start_date
        elif end_date.date() > last_cached_date:
            fetch_start_date = pd.Timestamp(last_cached_date) + timedelta(days=1)
        else:
            return df_cache[(df_cache['날짜'] >= pd.to_datetime(start_date)) & (df_cache['날짜'] <= pd.to_datetime(end_date))].reset_index(drop=True)

    new_data_list = []
    if fetch_start_date <= end_date:
        current_start = fetch_start_date
        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=89), end_date)
            all_data_chunk = []
            for keyword in keywords:
                body_search = json.dumps({
                    "startDate": current_start.strftime('%Y-%m-%d'),
                    "endDate": current_end.strftime('%Y-%m-%d'),
                    "timeUnit": "date",
                    "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
                })
                search_res = call_naver_api("https://openapi.naver.com/v1/datalab/search", body_search, naver_keys)
                if search_res and search_res.get('results') and search_res['results'][0]['data']:
                    df_search = pd.DataFrame(search_res['results'][0]['data'])
                    if not df_search.empty:
                         df_search.rename(columns={'period': '날짜', 'ratio': f'Naver_{keyword}'}, inplace=True)
                         all_data_chunk.append(df_search)
            if all_data_chunk:
                merged_chunk = reduce(lambda l, r: pd.merge(l, r, on='날짜', how='outer'), all_data_chunk)
                new_data_list.append(merged_chunk)
            current_start = current_end + timedelta(days=1)
    if new_data_list:
        df_new = pd.concat(new_data_list, ignore_index=True)
        df_new['날짜'] = pd.to_datetime(df_new['날짜'])
        if not df_cache.empty:
            df_combined = pd.concat([df_cache, df_new], ignore_index=True).sort_values('날짜').drop_duplicates(subset=['날짜'], keep='last')
        else:
            df_combined = df_new.sort_values('날짜')
        try:
            pandas_gbq.to_gbq(df_combined, f"{BQ_DATASET}.{BQ_TABLE_NAVER}", project_id=project_id, if_exists="replace", credentials=_client._credentials)
        except Exception as e:
            st.warning(f"Naver 트렌드 캐시 업데이트 실패: {e}")
        df_final = df_combined
    else:
        df_final = df_cache
    if df_final.empty:
        return pd.DataFrame()
    return df_final[(df_final['날짜'] >= pd.to_datetime(start_date)) & (df_final['날짜'] <= pd.to_datetime(end_date))].reset_index(drop=True)

# Initialize BQ Client and session state
bq_client = get_bq_connection()
if bq_client is None: st.stop()
if 'final_df' not in st.session_state: st.session_state.final_df = pd.DataFrame()

# ----------------------------
#  Sidebar: UI & Data Loading
# ----------------------------
st.sidebar.header("⚙️ 분석 설정")
categories = get_categories_from_bq(bq_client)
selected_categories = st.sidebar.multiselect("분석할 품목 선택", categories, default=categories[:1] if categories else [])
start_date = pd.to_datetime(st.sidebar.date_input('시작일', datetime(2022, 1, 1)))
end_date = pd.to_datetime(st.sidebar.date_input('종료일', datetime.now()))
news_keyword_input = st.sidebar.text_input("뉴스 분석 키워드", selected_categories[0] if selected_categories else "")
search_keywords_input = st.sidebar.text_input("네이버 트렌드 키워드 (쉼표로 구분)", ",".join(selected_categories) if selected_categories else "")
search_keywords = [k.strip() for k in search_keywords_input.split(',') if k.strip()]

with st.sidebar.expander("🔑 API 키 입력 (선택)"):
    naver_client_id = st.text_input("Naver API Client ID", type="password")
    naver_client_secret = st.text_input("Naver API Client Secret", type="password")
    st.sidebar.markdown("---")
    kamis_api_key = st.text_input("KAMIS API Key", type="password")
    kamis_api_id = st.text_input("KAMIS API ID", type="password")
    st.sidebar.markdown("##### KAMIS 농산물 가격 선택")
    kamis_item_name = st.sidebar.selectbox("품목 선택", list(KAMIS_FULL_DATA.keys()))
    kamis_kind_name = None
    if kamis_item_name:
        kamis_kind_name = st.sidebar.selectbox("품종 선택", list(KAMIS_FULL_DATA[kamis_item_name]['kinds'].keys()))

if st.sidebar.button("🚀 모든 데이터 통합 및 분석 실행"):
    if not selected_categories:
        st.error("분석할 품목을 1개 이상 선택해주세요.")
    else:
        with st.spinner("모든 데이터를 로드하고 통합하는 중입니다..."):
            trade_df = get_trade_data_from_bq(bq_client, selected_categories)
            trade_df_in_range = trade_df[(trade_df['Date'] >= start_date) & (trade_df['Date'] <= end_date)]
            
            if trade_df_in_range.empty:
                st.error("선택된 기간에 해당하는 데이터가 없습니다.")
            else:
                trade_agg = trade_df_in_range.groupby('Date').agg(Value=('Value', 'sum'), Volume=('Volume', 'sum')).copy()
                trade_agg.set_index(pd.to_datetime(trade_agg.index), inplace=True)
                trade_weekly = trade_agg.resample('W-Mon').agg(수입액_USD=('Value', 'sum'), 수입량_KG=('Volume', 'sum')).copy()
                trade_weekly['수입단가_USD_KG'] = trade_weekly['수입액_USD'] / trade_weekly['수입량_KG']
                trade_weekly.index.name = '날짜'
                
                all_weekly_dfs = {'trade': trade_weekly}

                if news_keyword_input:
                    news_df = get_news_with_multi_model_analysis(bq_client, models, news_keyword_input)
                    if not news_df.empty:
                        news_df['날짜'] = pd.to_datetime(news_df['날짜'])
                        sentiment_cols = [col for col in news_df.columns if 'Sentiment' in col]
                        news_weekly = news_df.set_index('날짜')[sentiment_cols].resample('W-Mon').mean()
                        all_weekly_dfs['news'] = news_weekly.rename(columns=lambda x: 'News_' + x.replace("News_", ""))

                if search_keywords and naver_client_id and naver_client_secret:
                    naver_df = fetch_naver_trends_data(bq_client, search_keywords, start_date, end_date, {'id': naver_client_id, 'secret': naver_client_secret})
                    if not naver_df.empty:
                        naver_df['날짜'] = pd.to_datetime(naver_df['날짜'])
                        naver_weekly = naver_df.set_index('날짜').resample('W-Mon').mean()
                        all_weekly_dfs['naver'] = naver_weekly

                if kamis_api_key and kamis_api_id and kamis_item_name and kamis_kind_name:
                    item_info = KAMIS_FULL_DATA[kamis_item_name].copy()
                    item_info['kind_code'] = item_info['kinds'][kamis_kind_name]
                    item_info['rank_code'] = '01'
                    kamis_df = fetch_kamis_data(bq_client, item_info, start_date, end_date, {'key': kamis_api_key, 'id': kamis_api_id})
                    if not kamis_df.empty:
                        kamis_df['날짜'] = pd.to_datetime(kamis_df['날짜'])
                        kamis_weekly = kamis_df.set_index('날짜').resample('W-Mon').mean()
                        all_weekly_dfs['kamis'] = kamis_weekly

                final_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_weekly_dfs.values())
                final_df = final_df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
                st.session_state.final_df = final_df.dropna(how='all', axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                st.success("데이터 통합 완료!")

# --- CSV/Excel Upload Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("품목 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 또는 Excel 파일을 업로드하세요.", type=['csv', 'xlsx'], help="필수 컬럼: Date, Value, Volume")
new_category_name = st.sidebar.text_input("업로드할 데이터의 품목명을 입력하세요.", help="예: 아보카도, 바나나 등")
if st.sidebar.button("파일을 BigQuery에 업로드"):
    if uploaded_file and new_category_name.strip():
        with st.spinner("파일 처리 및 업로드 중..."):
            try:
                df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, engine='openpyxl')
                df_upload.columns = [c.lower() for c in df_upload.columns]
                required_cols = {'date', 'value', 'volume'}
                if not required_cols.issubset(df_upload.columns):
                    st.sidebar.error("파일에 필수 컬럼('Date', 'Value', 'Volume')이 없습니다.")
                else:
                    df_upload = df_upload.rename(columns={'date': 'Date', 'value': 'Value', 'volume': 'Volume'})
                    df_upload['Date'] = pd.to_datetime(df_upload['Date'], errors='coerce')
                    df_upload['Value'] = pd.to_numeric(df_upload['Value'], errors='coerce')
                    df_upload['Volume'] = pd.to_numeric(df_upload['Volume'], errors='coerce')
                    df_upload['Category'] = new_category_name.strip()
                    df_to_bq = df_upload[['Date', 'Category', 'Value', 'Volume']].dropna()
                    success, error_msg = upload_df_to_bq(bq_client, df_to_bq, BQ_TABLE_TRADE)
                    if success:
                        st.sidebar.success(f"'{new_category_name}' 데이터 {len(df_to_bq)}건을 성공적으로 업로드했습니다!")
                        st.cache_data.clear()
                    else:
                        st.sidebar.error(f"업로드 실패: {error_msg}")
            except Exception as e:
                st.sidebar.error(f"파일 처리 오류: {e}")
    else:
        st.sidebar.warning("파일을 업로드하고 품목명을 입력해주세요.")

# ----------------------------
#  Guide Content
# ----------------------------
guide_content = """
### 🚀 대시보드 사용법 가이드
이 대시보드는 데이터 전문가가 아니더라도 누구나 쉽게 데이터 속 숨은 의미를 찾고 미래를 예측할 수 있도록 설계되었습니다. 각 기능의 의미와 활용법을 이해하면 더 강력한 인사이트를 얻을 수 있습니다.
---
### 1. 📊 상관관계 분석 탭 활용법
이 탭은 **"어떤 데이터들이 서로 관련이 있을까?"** 라는 질문에 답을 줍니다.
#### **① 분석 방법 선택: `Pearson` vs `Spearman`**
* **`Pearson (피어슨)`: 직선 관계를 보고 싶을 때** ("A가 2배 오르면 B도 정직하게 2배 오를까?")
* **`Spearman (스피어만)`: 방향성만 보고 싶을 때 (강력 추천!)** ("A가 오르면 B도 오르긴 오르나?")
> **💡 팁:** 헷갈린다면, 일반적으로 **`Spearman`**을 먼저 사용해 보세요.
#### **② 유의수준 (P-value) 필터 설정**
"이 관계가 우연일까, 진짜 의미가 있는 걸까?"를 판단하는 기준입니다.
* **`0.05` (기본값, 추천)**: 관계가 우연일 확률 5% 미만이라는 의미로, **"통계적으로 유의미하다"**고 판단합니다.
> **💡 팁:** 슬라이더를 `0.05`에 맞추고 여기서 나타나는 관계들을 '진짜' 신호로 해석하세요.
#### **③ 시차 교차상관 분석: 선행/후행 변수 선택**
"A의 변화가 몇 주 후에 B에게 영향을 미칠까?" 와 같은 **선후 관계**를 분석합니다.
* **`선행 변수 (Driver)`**: 원인이 되거나, 먼저 움직이는 변수. (예: `News_Sentiment`, `Naver_검색량`)
* **`후행 변수 (Outcome)`**: 영향을 받을 것으로 예상되는 결과 변수. (예: `수입단가_USD_KG`)
> **💡 결과 해석:** 'Best Lag (Weeks)'가 `2`로 나왔다면, 선행 변수가 2주 먼저 움직이고 2주 뒤에 후행 변수가 반응한다는 의미입니다.
---
### 2. 📈 시계열 예측 탭 활용법
이 탭은 **"그래서 미래에는 어떻게 될까?"** 라는 질문에 답을 줍니다.
#### **① 예측 대상 변수 (y) 선택**
가장 궁금한 **미래 값**을 선택합니다. (예: `수입단가_USD_KG`, `수입량_KG`)
#### **② Prophet 모델 파라미터 튜닝**
AI 예측 모델의 성격을 조절하여 우리 데이터에 꼭 맞게 만듭니다.
* **`Trend 유연성`**: 추세 변화에 얼마나 민감하게 반응할지 조절합니다. (시장이 역동적일수록 높게 설정)
* **`계절성 강도`**: 반복되는 패턴을 얼마나 강하게 믿을지 조절합니다. (계절성이 뚜렷할수록 높게 설정)
* **`계절성 모드`**: `multiplicative`는 비즈니스 성장에 따라 변동 폭도 함께 커질 때 사용합니다. (일반적으로 더 흔함)
> **💡 팁:** 처음에는 기본값으로 실행하고, 예측 그래프가 과거 데이터를 잘 못 따라갈 때 이 값들을 조절해보세요.
"""

# ----------------------------
#  Main Dashboard Tabs
# ----------------------------
if not st.session_state.final_df.empty:
    final_df = st.session_state.final_df
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 상관관계 분석", "📈 시계열 예측", "📄 통합 데이터", "📘 대시보드 사용법"])

    with tab1:
        st.header("상관관계 분석")
        col1, col2 = st.columns(2)
        corr_method = col1.selectbox("상관관계 분석 방법", ('pearson', 'spearman'), help="피어슨: 선형, 스피어만: 비선형 관계")
        pval_threshold = col2.slider("유의수준(P-value) 필터", 0.0, 1.0, 0.05)
        corr_matrix, pval_matrix = calculate_advanced_correlation(final_df, method=corr_method)
        st.subheader(f"'{corr_method.capitalize()}' 상관관계 히트맵")
        fig_heatmap = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu_r', zmin=-1, zmax=1, text=corr_matrix.round(2).astype(str), texttemplate="%{text}"))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        with st.expander("🔍 히트맵 결과 해석하기 (Driver vs Outcome)"):
            st.markdown(interpret_correlation_groups(corr_matrix, pval_matrix, threshold=pval_threshold))
        st.markdown("---")
        st.subheader("시차 교차상관 분석")
        driver_cols = st.multiselect("선행 변수(Driver) 선택", final_df.columns, default=[c for c in final_df if any(k in c for k in ['News', 'Naver', '도매'])])
        outcome_cols = st.multiselect("후행 변수(Outcome) 선택", final_df.columns, default=[c for c in final_df if '수입' in c])
        if driver_cols and outcome_cols:
            lag_df = find_best_lagged_correlation(final_df, driver_cols, outcome_cols)
            st.dataframe(lag_df.head(10))
            if not lag_df.empty:
                top_lag = lag_df.iloc[0]
                with st.expander("🔍 시차 분석 결과 해석하기"):
                    st.info(f"가장 강한 시차 관계: **{top_lag['Driver (X)']}**의 변화는 **{top_lag['Best Lag (Weeks)']}주 후** **{top_lag['Outcome (Y)']}**에 영향을 미치는 경향이 있습니다 (상관계수: {top_lag['Correlation']:.3f}).")

    with tab2:
        st.header("시계열 예측 (Prophet & XGBoost)")
        prophet_df = final_df.reset_index().rename(columns={'날짜': 'ds'})
        col1, col2 = st.columns(2)
        forecast_col = col1.selectbox("예측 대상 변수 (y)", final_df.columns)
        forecast_periods = col2.number_input("예측 기간 (주)", 4, 52, 12)
        prophet_df = prophet_df.rename(columns={forecast_col: 'y'})
        regressors = [c for c in prophet_df.columns if c not in ['ds', 'y']]
        selected_regressors = st.multiselect("외부 예측 변수", regressors, default=regressors)
        st.subheader("Prophet 모델 파라미터 튜닝")
        p_col1, p_col2, p_col3 = st.columns(3)
        changepoint_prior_scale = p_col1.slider("Trend 유연성", 0.01, 0.5, 0.05)
        seasonality_prior_scale = p_col2.slider("계절성 강도", 0.01, 10.0, 1.0)
        seasonality_mode = p_col3.selectbox("계절성 모드", ('additive', 'multiplicative'))
        if st.button("🚀 예측 실행", key="run_forecast"):
            with st.spinner("Prophet & XGBoost 모델 학습 및 예측 중..."):
                m = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale, seasonality_mode=seasonality_mode)
                for reg in selected_regressors: m.add_regressor(reg)
                m.fit(prophet_df[['ds', 'y'] + selected_regressors])
                future = m.make_future_dataframe(periods=forecast_periods, freq='W')
                future_regressors = prophet_df[['ds'] + selected_regressors].set_index('ds')
                last_values = future_regressors.iloc[-1]
                future_regressors = future_regressors.reindex(future['ds']).fillna(method='ffill').fillna(last_values)
                future = pd.concat([future.set_index('ds'), future_regressors], axis=1).reset_index()
                forecast = m.predict(future)
                ml_df = forecast[['ds', 'trend', 'yearly', 'weekly']].set_index('ds').join(prophet_df.set_index('ds')).dropna()
                X, y = ml_df.drop(columns=['y']), ml_df['y']
                train_size = int(len(X) * 0.85)
                X_train, X_test, y_train, y_test = X.iloc[:train_size], X.iloc[train_size:], y.iloc[:train_size], y.iloc[train_size:]
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, early_stopping_rounds=50)
                xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                y_pred_xgb = xgb_model.predict(X_test)
                r2, rmse = r2_score(y_test, y_pred_xgb), np.sqrt(mean_squared_error(y_test, y_pred_xgb))
                feature_imp = pd.DataFrame(sorted(zip(xgb_model.feature_importances_, X.columns)), columns=['Value','Feature'])
            st.subheader("Prophet 예측 결과")
            st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
            with st.expander("🔍 Prophet 예측 해석"): st.markdown("- **검은 점:** 실제 데이터\n- **파란선:** 예측값\n- **파란 영역:** 불확실성 구간")
            st.subheader("Prophet 요인 분해")
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)
            st.markdown("---")
            st.subheader("모델 진단: 잔차 분석")
            df_pred = forecast.set_index('ds')[['yhat']].join(prophet_df.set_index('ds')[['y']]).dropna()
            residuals = df_pred['y'] - df_pred['yhat']
            diag_col1, diag_col2 = st.columns(2)
            adf_result = adfuller(residuals)
            diag_col1.metric("잔차 ADF Test p-value", f"{adf_result[1]:.4f}")
            with diag_col1.expander("🔍 ADF 테스트 해석"): st.markdown(interpret_adf_test(adf_result))
            fig_dist = ff.create_distplot([residuals], ['residuals'], show_rug=False)
            diag_col2.plotly_chart(fig_dist, use_container_width=True)
            st.markdown("---")
            st.subheader("XGBoost Meta-Forecasting 결과")
            xgb_col1, xgb_col2 = st.columns(2)
            xgb_col1.metric("Test R² Score", f"{r2:.3f}")
            xgb_col2.metric("Test RMSE", f"{rmse:.3f}")
            fig_xgb = go.Figure()
            fig_xgb.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Actual'))
            fig_xgb.add_trace(go.Scatter(x=y_test.index, y=y_pred_xgb, name='XGBoost Prediction'))
            st.plotly_chart(fig_xgb, use_container_width=True)
            st.plotly_chart(px.bar(feature_imp, x="Value", y="Feature", orientation='h', title="Feature Importance"), use_container_width=True)
            with st.expander("🔍 XGBoost 결과 해석"): st.markdown(interpret_xgboost_results(r2, rmse, feature_imp, y_test.mean()))

    with tab3:
        st.header("통합 데이터 (주별)")
        st.dataframe(final_df)
        st.download_button("CSV로 다운로드", final_df.to_csv(index=False).encode('utf-8-sig'), "integrated_weekly_data.csv")
        
    with tab4:
        st.header("대시보드 사용법 가이드")
        st.markdown(guide_content)

else:
    st.info("👈 사이드바에서 분석할 데이터를 선택하고 '모든 데이터 통합 및 분석 실행' 버튼을 눌러주세요.")

