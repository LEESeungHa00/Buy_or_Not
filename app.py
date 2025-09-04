import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, timezone
from functools import reduce
import json
import urllib.request
import yfinance as yf
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas_gbq
from newspaper import Article, Config
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.plot import plot_plotly
import feedparser
from urllib.parse import quote
import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import logging
from google.cloud import language_v1

# --- Constants & Global Settings ---
BQ_TABLE_NEWS = "news_sentiment_analysis_results" # 뉴스 분석 결과를 저장할 테이블 이름
BQ_TABLE_NAVER = "naver_trends_cache" # [추가] 네이버 트렌드 캐시 테이블 이름
# GPU 사용 설정 (Streamlit Cloud에서는 CPU를 사용하게 됩니다)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
bq_client = get_bq_connection()
nlp_client = get_gcp_nlp_client() # NLP 클라이언트 추가
if bq_client is None or nlp_client is None: st.stop()
    
# BigQuery 설정
BQ_DATASET = "data_explorer"  # 데이터를 저장할 BigQuery 데이터셋 이름
BQ_TABLE_NEWS = "news_sentiment_analysis_results" # 뉴스 분석 결과를 저장할 테이블 이름

# 감성 분석 모델 상세 설정
BATCH_SIZE = 8                # 한 번에 몇 개의 기사를 분석할지 결정
TOPK_WORDS = 5                # 긍정/부정 키워드를 몇 개까지 보여줄지 결정
IG_PROB_THRESH = 0.7          # 예측 확률이 이 값보다 낮으면 상세 분석(IG) 실행
IG_MARGIN_THRESH = 0.2        # 1위 예측과 2위 예측 확률 차이가 이 값보다 작으면 상세 분석(IG) 실행
IG_N_STEPS = 50               # 상세 분석(IG)의 정확도 관련 설정

# 한국어 형태소 분석기 설정 (Streamlit Cloud 설치가 까다로우므로 우선 False로 둡니다)
MECAB_AVAILABLE = False
# try:
#     from konlpy.tag import Mecab
#     mecab = Mecab()
#     MECAB_AVAILABLE = True
# except ImportError:
#     MECAB_AVAILABLE = False

# --- BigQuery Connection ---
@st.cache_resource
def get_bq_connection():
    """BigQuery에 직접 연결하고 클라이언트 객체를 반환합니다."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        client = bigquery.Client(credentials=creds, project=creds.project_id)
        return client
    except Exception as e:
        st.error(f"Google BigQuery 연결 실패: secrets.toml 설정을 확인하세요. 오류: {e}")
        return None
        
def ensure_bq_table_schema(client, dataset_id, table_id):
    """BigQuery에 테이블이 없으면 지정된 스키마로 생성합니다."""
    project_id = client.project
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    try:
        client.get_table(full_table_id)
        # st.write(f"테이블 '{full_table_id}'은 이미 존재합니다.")
    except Exception:
        st.write(f"테이블 '{full_table_id}'을 새로 생성합니다.")
        schema = [
            bigquery.SchemaField("날짜", "DATE"),
            bigquery.SchemaField("Title", "STRING"),
            bigquery.SchemaField("Label", "STRING"),
            bigquery.SchemaField("Prob", "FLOAT"),
            bigquery.SchemaField("Top_Positive_Keywords", "STRING"),
            bigquery.SchemaField("Top_Negative_Keywords", "STRING"),
            bigquery.SchemaField("Keyword", "STRING"),
            bigquery.SchemaField("Language", "STRING"),
            bigquery.SchemaField("Sentiment", "FLOAT"),
            bigquery.SchemaField("InsertedAt", "TIMESTAMP"),
        ]
        table = bigquery.Table(full_table_id, schema=schema)
        client.create_table(table)
        st.success(f"테이블 '{table_id}' 생성 완료.")
        
# --- Sentiment Models ---
# --- Google NLP Client ---
@st.cache_resource
def get_gcp_nlp_client():
    """Google Cloud Natural Language API 클라이언트를 반환합니다."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        nlp_client = language_v1.LanguageServiceClient(credentials=creds)
        return nlp_client
    except Exception as e:
        st.error(f"Google NLP 연결 실패: {e}")
        return None

def analyze_sentiment_with_google(text_content, nlp_client):
    """주어진 텍스트의 감성을 Google NLP API를 사용해 분석합니다."""
    if not text_content or not nlp_client:
        return 0.0, 0.0

    document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = nlp_client.analyze_sentiment(request={'document': document})
    return response.document_sentiment.score, response.document_sentiment.magnitude

def fetch_and_analyze_news_lightweight(bq_client, nlp_client, keyword, days_limit=7):
    """
    뉴스를 수집하고 Google NLP로 분석 후, BigQuery에 캐싱하는 경량화된 함수.
    """
    project_id = bq_client.project
    dataset_id = BQ_DATASET
    table_id = BQ_TABLE_NEWS
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    # 1. BigQuery에서 최신 캐시 확인
    try:
        time_limit = datetime.now(timezone.utc) - timedelta(days=days_limit)
        query = f"""
            SELECT * FROM `{full_table_id}`
            WHERE Keyword = @keyword AND InsertedAt >= @time_limit
            ORDER BY 날짜 DESC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("keyword", "STRING", keyword),
                bigquery.ScalarQueryParameter("time_limit", "TIMESTAMP", time_limit),
            ]
        )
        df_cache = bq_client.query(query, job_config=job_config).to_dataframe()
    except Exception:
        df_cache = pd.DataFrame()

    if not df_cache.empty:
        st.sidebar.success(f"✔️ '{keyword}' 최신 분석 결과를 캐시에서 로드했습니다.")
        return df_cache

    # 2. 캐시 없으면 새로 수집 및 분석
    st.sidebar.warning(f"'{keyword}'에 대한 최신 캐시가 없습니다.\n새로 뉴스를 분석합니다...")
    
    with st.spinner(f"'{keyword}' 뉴스 수집 및 분석 중..."):
        all_news = []
        rss_url = f"https://news.google.com/rss/search?q={quote(keyword)}&hl=ko&gl=KR&ceid=KR:ko"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:20]: # 기사 20개로 제한
            title = entry.get('title', '')
            if not title: continue
            try:
                pub_date = pd.to_datetime(entry.get('published')).date()
                all_news.append({"날짜": pub_date, "Title": title})
            except:
                continue
        
        if not all_news:
            st.error("분석할 뉴스를 찾지 못했습니다.")
            return pd.DataFrame()

        df_new = pd.DataFrame(all_news)
        df_new = df_new.drop_duplicates(subset=["Title"])

        # Google NLP API로 감성 분석 실행
        sentiments = [analyze_sentiment_with_google(title, nlp_client) for title in df_new['Title']]
        df_new['Sentiment'] = [s[0] for s in sentiments] # score
        df_new['Magnitude'] = [s[1] for s in sentiments] # magnitude
        df_new['Keyword'] = keyword
        df_new['InsertedAt'] = datetime.now(timezone.utc)
        
        # 3. BigQuery에 새 결과 저장
        if not df_new.empty:
            st.sidebar.info("새 분석 결과를 BigQuery 캐시에 저장합니다.")
            # 테이블 스키마에 맞게 컬럼 정리 (Label 등 불필요한 컬럼 제외)
            final_cols = {
                "날짜": "DATE", "Title": "STRING", "Keyword": "STRING", 
                "Sentiment": "FLOAT", "Magnitude": "FLOAT", "InsertedAt": "TIMESTAMP"
            }
            df_to_gbq = df_new[list(final_cols.keys())]
            pandas_gbq.to_gbq(df_to_gbq, full_table_id, project_id=project_id, if_exists="append")
        
        return df_to_gbq
        
# --- Helper functions ---
def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def predict_batch(texts, model, tokenizer, device=DEVICE, max_len=128):
    model.eval()
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                    max_length=max_len, return_offsets_mapping=False)
    if device != "cpu":
        enc = {k: v.to(device) for k, v in enc.items()}
        model.to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = _softmax(logits.cpu().numpy())
    id2label = model.config.id2label
    labels = [id2label[i].lower() for i in range(logits.shape[-1])]
    return probs, labels

def compute_margin_and_pred(probs, labels):
    order = np.argsort(probs)[::-1]
    pred_idx = int(order[0])
    pred_prob = float(probs[pred_idx])
    second = float(probs[order[1]]) if len(order) > 1 else 0.0
    margin = pred_prob - second
    return pred_idx, pred_prob, margin

class ForwardWrapper(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def token_attributions_ig(text, model, tokenizer, target_idx,
                          max_len=128, n_steps=IG_N_STEPS):
    model.eval()
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length',
                    max_length=max_len, return_offsets_mapping=True)
    input_ids, attention_mask = enc['input_ids'], enc['attention_mask']
    offsets = enc['offset_mapping'][0].tolist()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baselines = torch.full_like(input_ids, pad_id)
    fw = ForwardWrapper(model)
    lig = LayerIntegratedGradients(fw, model.get_input_embeddings())
    attributions, _ = lig.attribute(inputs=input_ids,
                                    baselines=baselines,
                                    additional_forward_args=(attention_mask,),
                                    target=target_idx,
                                    n_steps=n_steps)
    token_attr = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
    token_attr = token_attr * attention_mask.squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    keep_idx = [i for i, t in enumerate(tokens)
                if t not in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token)]
    words = []
    for i in keep_idx:
        s, e = offsets[i]
        word = text[s:e]
        words.append((word, float(token_attr[i])))
    if MECAB_AVAILABLE:
        merged = {}
        for w, sc in words:
            try: morphs = mecab.nouns(w) or [w]
            except: morphs = [w]
            for m in morphs: merged[m] = merged.get(m, 0.0) + sc
        words = list(merged.items())
    pos = sorted([x for x in words if x[1] > 0], key=lambda x: -x[1])
    neg = sorted([x for x in words if x[1] < 0], key=lambda x: x[1])
    return pos, neg


# --- Data Fetching & Processing Functions ---

@st.cache_data(ttl=3600)
def get_categories_from_bq(_client):
    """BigQuery에서 분석 가능한 품목 카테고리 목록을 가져옵니다."""
    project_id = _client.project
    table_id = f"{project_id}.data_explorer.tds_data"
    try:
        with st.spinner("BigQuery에서 카테고리 목록 불러오는 중..."):
            query = f"SELECT DISTINCT Category FROM `{table_id}` WHERE Category IS NOT NULL ORDER BY Category"
            df = _client.query(query).to_dataframe()
        return sorted(df['Category'].astype(str).unique())
    except Exception as e:
        st.error(f"BigQuery 테이블({table_id})을 읽는 중 오류: {e}")
        return []

def get_trade_data_from_bq(client, categories):
    """선택된 카테고리에 대한 수출입 데이터를 BigQuery에서 가져옵니다."""
    if not categories: return pd.DataFrame()
    project_id = client.project
    table_id = f"{project_id}.data_explorer.tds_data"
    try:
        query_params = [bigquery.ArrayQueryParameter("categories", "STRING", categories)]
        sql = f"SELECT * FROM `{table_id}` WHERE Category IN UNNEST(@categories)"
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        with st.spinner(f"BigQuery에서 {len(categories)}개 카테고리 데이터 로드 중..."):
            df = client.query(sql, job_config=job_config).to_dataframe()
        
        # 데이터 타입 정리
        for col in df.columns:
            if 'price' in col.lower() or 'value' in col.lower() or 'volume' in col.lower():
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"BigQuery에서 TDS 데이터 읽는 중 오류: {e}")
        return pd.DataFrame()

def deduplicate_and_write_to_bq(client, df_new, table_name, subset_cols=None):
    """데이터를 BigQuery에 중복 제거하여 저장합니다."""
    project_id = client.project
    table_id = f"{project_id}.data_explorer.{table_name}"
    try:
        try:
            sql = f"SELECT * FROM `{table_id}`"
            df_existing = client.query(sql).to_dataframe()
        except Exception: 
            df_existing = pd.DataFrame()

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        if subset_cols:
            df_deduplicated = df_combined.drop_duplicates(subset=subset_cols, keep='last')
        else:
            df_deduplicated = df_combined.drop_duplicates(keep='last')
        
        with st.spinner(f"'{table_name}' 테이블에 데이터 저장 중..."):
            pandas_gbq.to_gbq(df_deduplicated, table_id, project_id=project_id, if_exists="replace", credentials=client._credentials)
        st.sidebar.success(f"'{table_name}'에 데이터 업데이트 완료.")
    except Exception as e: 
        st.error(f"BigQuery 저장 중 오류: {e}")

def call_naver_api(url, body, naver_keys):
    """네이버 API를 호출하고 결과를 반환합니다."""
    try:
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", naver_keys['id'])
        request.add_header("X-Naver-Client-Secret", naver_keys['secret'])
        request.add_header("Content-Type", "application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        if response.getcode() == 200:
            return json.loads(response.read().decode('utf-8'))
        return None
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        st.error(f"Naver API 오류 발생: {e.code} - {e.reason}")
        st.error(f"서버 응답: {error_body}")
        return None
    except Exception as e:
        st.error(f"API 호출 중 알 수 없는 오류 발생: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_naver_trends_data(_client, keywords, start_date, end_date, naver_keys):
    """
    BigQuery 캐시를 '누적'하여 활용하는 진짜 캐싱 기능으로
    3개월 이상 긴 기간의 데이터를 효율적으로 가져옵니다.
    """
    project_id = _client.project
    dataset_id = BQ_DATASET
    table_id = BQ_TABLE_NAVER
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    # 1. BigQuery에서 기존 캐시 데이터 읽기
    try:
        sql = f"SELECT * FROM `{full_table_id}` ORDER BY 날짜"
        df_cache = _client.query(sql).to_dataframe()
        if not df_cache.empty:
            df_cache['날짜'] = pd.to_datetime(df_cache['날짜'])
    except Exception:
        df_cache = pd.DataFrame(columns=['날짜'])

    # 2. 캐시를 바탕으로 API를 호출할 새로운 기간 결정
    fetch_start_date = start_date
    if not df_cache.empty:
        last_cached_date = df_cache['날짜'].max()
        if start_date > last_cached_date:
            fetch_start_date = start_date
        elif end_date > last_cached_date:
            fetch_start_date = last_cached_date + timedelta(days=1)
        else: # 요청 기간 전체가 캐시에 있는 경우
            fetch_start_date = end_date + timedelta(days=1) # API 호출 안 하도록 설정

    new_data_list = []
    if fetch_start_date <= end_date:
        st.sidebar.write(f"새로운 데이터 수집: {fetch_start_date.date()} ~ {end_date.date()}")
        # ... (이하 API 호출 로직은 이전과 동일)
        current_start = fetch_start_date
        while current_start <= end_date:
            current_end = current_start + timedelta(days=89)
            if current_end > end_date:
                current_end = end_date

            st.sidebar.info(f"Naver API 호출 중: {current_start.date()} ~ {current_end.date()}")
            
            NAVER_SHOPPING_CAT_MAP = {
                '아보카도': "50000007", '바나나': "50000007", '사과': "50000007", '수입과일': "50000007",
                '커피': "50000004", '커피 생두': "50000004",
                '쌀': "50000006", '고등어': "50000009"
            }
            all_data_chunk = []
            for keyword in keywords:
                keyword_dfs = []
                body_search = json.dumps({
                    "startDate": current_start.strftime('%Y-%m-%d'), "endDate": current_end.strftime('%Y-%m-%d'), "timeUnit": "date", "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}]
                })
                search_res = call_naver_api("https://openapi.naver.com/v1/datalab/search", body_search, naver_keys)
                if search_res and search_res.get('results') and search_res['results'][0]['data']:
                    df_search = pd.DataFrame(search_res['results'][0]['data'])
                    if not df_search.empty and 'period' in df_search.columns:
                        keyword_dfs.append(df_search.rename(columns={'period': '날짜', 'ratio': f'NaverSearch_{keyword}'}))
                
                lower_keyword = keyword.lower().replace(' ', '')
                if lower_keyword in NAVER_SHOPPING_CAT_MAP:
                    category_id = NAVER_SHOPPING_CAT_MAP[lower_keyword]
                    body_shop = json.dumps({
                        "startDate": current_start.strftime('%Y-%m-%d'),"endDate": current_end.strftime('%Y-%m-%d'), "timeUnit": "date",
                        "category": [{"name": keyword, "param": [category_id]}], "keyword": [{"name": keyword, "param": [keyword]}]
                    })
                    shop_res = call_naver_api("https://openapi.naver.com/v1/datalab/shopping/categories", body_shop, naver_keys)
                    if shop_res and shop_res.get('results') and shop_res['results'][0]['data']:
                        df_shop = pd.DataFrame(shop_res['results'][0]['data'])
                        if not df_shop.empty and 'period' in df_shop.columns:
                            keyword_dfs.append(df_shop.rename(columns={'period': '날짜', 'ratio': f'NaverShop_{keyword}'}))

                if keyword_dfs:
                    for df in keyword_dfs: df['날짜'] = pd.to_datetime(df['날짜'])
                    merged_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), keyword_dfs)
                    all_data_chunk.append(merged_df)
            
            if all_data_chunk:
                chunk_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), all_data_chunk)
                new_data_list.append(chunk_df)

            current_start = current_end + timedelta(days=1)
    else:
        st.sidebar.success("✔️ 요청 기간의 모든 데이터가 캐시에 있습니다.")


    # 4. 새로 가져온 데이터가 있다면, 캐시에 '추가'하고 기존 데이터와 통합
 # 4. 새로 가져온 데이터가 있다면, 캐시에 '추가'하고 기존 데이터와 통합
    if new_data_list:
        # [핵심 수정] 리스트에서 비어있지 않은 데이터프레임만 필터링합니다.
        non_empty_dfs = [df for df in new_data_list if not df.empty]
        
        # 필터링 후에도 데이터가 남아있을 때만 concat을 실행합니다.
        if non_empty_dfs:
            df_new = pd.concat(non_empty_dfs, ignore_index=True)
            df_new['날짜'] = pd.to_datetime(df_new['날짜'])
        else:
            df_new = pd.DataFrame() # 모든 API 호출 결과가 비어있었을 경우
        
        # [핵심 수정] 기존 캐시와 합치고, 중복 제거 후, 'replace'가 아닌 전체를 다시 쓰기
        # (BigQuery는 기본 append가 까다로워, 읽고-합치고-전체쓰기 방식이 안정적)
        df_combined = pd.concat([df_cache, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=['날짜'], keep='last', inplace=True)
        df_combined = df_combined.sort_values(by='날짜').reset_index(drop=True)

        st.sidebar.info(f"'{table_id}' 캐시 테이블 업데이트 중...")
        pandas_gbq.to_gbq(df_combined, full_table_id, project_id=project_id, if_exists="replace", credentials=_client._credentials)
        st.sidebar.success("캐시 업데이트 완료.")
        df_final = df_combined
    else:
        df_final = df_cache

    if df_final.empty:
        return pd.DataFrame()
    
    # 5. 사용자가 요청한 기간에 맞춰 최종 데이터 반환
    return df_final[(df_final['날짜'] >= start_date) & (df_final['날짜'] <= end_date)].reset_index(drop=True) 

def fetch_kamis_data(client, item_info, start_date, end_date, kamis_keys):
    """KAMIS에서 기간별 도매 가격 데이터를 한 번의 API 호출로 가져옵니다."""
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    item_code, kind_code = item_info['item_code'], item_info['kind_code']

    with st.spinner("KAMIS에서 기간 데이터 조회 중..."):
        url = (
            "http://www.kamis.or.kr/service/price/xml.do?action=periodWholesaleProductList"
            f"&p_product_cls_code=01&p_startday={start_str}&p_endday={end_str}"
            f"&p_item_category_code={item_info['cat_code']}&p_item_code={item_code}&p_kind_code={kind_code}"
            f"&p_product_rank_code={item_info['rank_code']}&p_convert_kg_yn=Y"
            f"&p_cert_key={kamis_keys['key']}&p_cert_id={kamis_keys['id']}&p_returntype=json"
        )
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if data and "data" in data and data["data"] and "item" in data["data"]:
                    price_data = data["data"]["item"]
                    if not price_data:
                        st.sidebar.warning("해당 기간에 대한 KAMIS 데이터가 없습니다.")
                        return pd.DataFrame()
                    
                    df_new = pd.DataFrame(price_data).rename(columns={'regday': '날짜', 'price': '도매가격_원'})
                    df_new = df_new[['날짜', '도매가격_원']]
                    
                    def format_date(date_str):
                        return f"{start_date.year}-{date_str}" if len(date_str) <= 5 else date_str
                    
                    df_new['날짜'] = pd.to_datetime(df_new['날짜'].apply(format_date))
                    df_new['도매가격_원'] = pd.to_numeric(df_new['도매가격_원'].str.replace(',', ''), errors='coerce')
                    return df_new
                else:
                    st.sidebar.warning("해당 기간에 대한 KAMIS 데이터가 없습니다.")
            else:
                st.sidebar.error(f"KAMIS 서버 응답 오류: Status {response.status_code}")

        except Exception as e:
            st.sidebar.error(f"KAMIS API 호출 중 오류: {e}")

    return pd.DataFrame()


# --- Constants & App ---
KAMIS_FULL_DATA = {
    '쌀': {'cat_code': '100', 'item_code': '111', 'kinds': {'20kg': '01', '백미': '02'}},
    '감자': {'cat_code': '100', 'item_code': '152', 'kinds': {'수미(노지)': '01', '수미(시설)': '04'}},
    '배추': {'cat_code': '200', 'item_code': '211', 'kinds': {'봄': '01', '여름': '02', '가을': '03'}},
    '양파': {'cat_code': '200', 'item_code': '245', 'kinds': {'양파': '00', '햇양파': '02'}},
    '사과': {'cat_code': '400', 'item_code': '411', 'kinds': {'후지': '05', '아오리': '06'}},
    '바나나': {'cat_code': '400', 'item_code': '418', 'kinds': {'수입': '02'}},
    '아보카도': {'cat_code': '400', 'item_code': '430', 'kinds': {'수입': '00'}},
    '고등어': {'cat_code': '600', 'item_code': '611', 'kinds': {'생선': '01', '냉동': '02'}},
}
st.set_page_config(layout="wide")
st.title("📊 데이터 탐색 및 통합 분석 대시보드")

bq_client = get_bq_connection()
sentiment_assets = load_sentiment_assets()
if bq_client is None: st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ 분석 설정")
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'categories' not in st.session_state: st.session_state.categories = get_categories_from_bq(bq_client)

st.sidebar.subheader("1. 분석 대상 설정")
if not st.session_state.categories:
    st.sidebar.warning("BigQuery에 데이터가 없습니다. 새 데이터를 추가해주세요.")
else:
    selected_categories = st.sidebar.multiselect("분석할 품목 카테고리 선택", st.session_state.categories, default=st.session_state.get('selected_categories', []))
    if st.sidebar.button("🚀 선택 완료 및 분석 시작"):
        if not selected_categories:
            st.sidebar.warning("카테고리를 선택해주세요.")
        else:
            st.session_state.raw_trade_df = get_trade_data_from_bq(bq_client, selected_categories)
            if st.session_state.raw_trade_df is not None and not st.session_state.raw_trade_df.empty:
                st.session_state.data_loaded = True
                st.session_state.selected_categories = selected_categories
            else:
                st.sidebar.error("데이터를 불러오지 못했습니다.")

if not st.session_state.data_loaded:
    st.info("👈 사이드바에서 분석할 카테고리를 선택하고 '분석 시작' 버튼을 눌러주세요.")
    st.stop()

# --- Main App Logic ---
raw_trade_df = st.session_state.raw_trade_df
selected_categories = st.session_state.selected_categories
st.sidebar.success(f"**{', '.join(selected_categories)}** 데이터 로드 완료!")
st.sidebar.markdown("---")

try:
    raw_trade_df.dropna(subset=['Date'], inplace=True)
    file_start_date, file_end_date = raw_trade_df['Date'].min(), raw_trade_df['Date'].max()
    
    st.sidebar.subheader("2. 분석 기간 및 키워드 설정")
    start_date = pd.to_datetime(st.sidebar.date_input('시작일', file_start_date, min_value=file_start_date, max_value=file_end_date))
    end_date = pd.to_datetime(st.sidebar.date_input('종료일', file_end_date, min_value=start_date, max_value=file_end_date))
    
    default_keywords = ", ".join(selected_categories)
    keyword_input = st.sidebar.text_input("뉴스/트렌드 분석 키워드", default_keywords)
    search_keywords = [k.strip() for k in keyword_input.split(',') if k.strip()]
except Exception as e:
    st.error(f"데이터 기간 설정 중 오류: {e}")
    st.stop()

st.sidebar.subheader("3. 외부 데이터 연동")

st.sidebar.markdown("##### KAMIS 농산물 가격")
kamis_api_key = st.sidebar.text_input("KAMIS API Key", type="password", help="KAMIS 공공데이터 포털에서 발급받은 인증키")
kamis_api_id = st.sidebar.text_input("KAMIS API ID", type="password", help="KAMIS API 신청 시 등록한 ID")
item_name = st.sidebar.selectbox("품목 선택", list(KAMIS_FULL_DATA.keys()))
if item_name:
    kind_name = st.sidebar.selectbox("품종 선택", list(KAMIS_FULL_DATA[item_name]['kinds'].keys()))
    if st.sidebar.button("KAMIS 데이터 가져오기"):
        if kamis_api_key and kamis_api_id:
            item_info = {**KAMIS_FULL_DATA[item_name], 'item_name': item_name, 'kind_name': kind_name, 'rank_code': '01'}
            item_info['kind_code'] = KAMIS_FULL_DATA[item_name]['kinds'][kind_name]
            st.session_state.wholesale_data = fetch_kamis_data(bq_client, item_info, start_date, end_date, {'key': kamis_api_key, 'id': kamis_api_id})
        else:
            st.sidebar.error("KAMIS API Key와 ID를 모두 입력해주세요.")

st.sidebar.markdown("##### 네이버 트렌드 데이터")
naver_client_id = st.sidebar.text_input("Naver API Client ID", type="password")
naver_client_secret = st.sidebar.text_input("Naver API Client Secret", type="password")
if st.sidebar.button("네이버 트렌드 가져오기"):
    if not search_keywords: st.sidebar.warning("분석할 키워드를 먼저 입력해주세요.")
    else:
        # bq_client를 첫 번째 인자로 추가
        st.session_state.search_data = fetch_naver_trends_data(bq_client, search_keywords, start_date, end_date, {'id': naver_client_id, 'secret': naver_client_secret})

st.sidebar.markdown("##### 뉴스 감성 분석")
news_keyword = st.sidebar.text_input("분석할 뉴스 키워드 입력", placeholder="예: 커피")

if st.sidebar.button("📰 뉴스 감성 분석 실행"):
    if not news_keyword:
        st.sidebar.warning("분석할 키워드를 먼저 입력해주세요.")
    else:
        # 새로 만든 경량화 함수 호출
        result_df = fetch_and_analyze_news_lightweight(bq_client, nlp_client, news_keyword)
        st.session_state.news_data = result_df
        
# --- Main Display Tabs ---
raw_wholesale_df = st.session_state.get('wholesale_data', pd.DataFrame())
raw_search_df = st.session_state.get('search_data', pd.DataFrame())
raw_news_df = st.session_state.get('news_data', pd.DataFrame())

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1️⃣ 원본 데이터", "2️⃣ 데이터 표준화", "3️⃣ 뉴스 감성 분석", "4️⃣ 상관관계 분석", "📈 시계열 예측"])

with tab1:
    st.subheader("A. 수출입 데이터 (선택 기간)"); st.dataframe(raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)])
    st.subheader("B. 외부 가격 데이터"); st.dataframe(raw_wholesale_df)
    st.subheader("C. 트렌드 데이터"); st.dataframe(raw_search_df)
    st.subheader("D. 뉴스 데이터"); st.dataframe(raw_news_df)
    
with tab2:
    st.header("데이터 표준화: 주별(Weekly) 데이터로 변환")
    trade_df_in_range = raw_trade_df[(raw_trade_df['Date'] >= start_date) & (raw_trade_df['Date'] <= end_date)]
    filtered_trade_df = trade_df_in_range[trade_df_in_range['Category'].isin(selected_categories)].copy()
    
    if not filtered_trade_df.empty:
        filtered_trade_df.set_index('Date', inplace=True)
        trade_weekly = filtered_trade_df.resample('W-Mon').agg(수입액_USD=('Value', 'sum'), 수입량_KG=('Volume', 'sum')).copy()
        trade_weekly['수입단가_USD_KG'] = trade_weekly['수입액_USD'] / trade_weekly['수입량_KG']
        trade_weekly.index.name = '날짜'
        
        dfs_to_process = {
            'wholesale': raw_wholesale_df,
            'search': raw_search_df,
            'news': raw_news_df
        }
        weekly_dfs = {'trade': trade_weekly}

        for name, df in dfs_to_process.items():
            if not df.empty and '날짜' in df.columns:
                df['날짜'] = pd.to_datetime(df['날짜'])
                df_in_range = df[(df['날짜'] >= start_date) & (df['날짜'] <= end_date)]
                if not df_in_range.empty:
                    df_weekly = df_in_range.set_index('날짜').resample('W-Mon').mean(numeric_only=True)
                    df_weekly.index.name = '날짜'
                    weekly_dfs[name] = df_weekly

        st.session_state['weekly_dfs'] = weekly_dfs
        st.write("### 주별 집계 데이터 샘플")
        for name, df in weekly_dfs.items():
            st.write(f"##### {name.capitalize()} Data (Weekly)")
            st.dataframe(df.head())
    else:
        st.warning("선택된 기간에 해당하는 수출입 데이터가 없습니다.")

with tab3:
    st.header("뉴스 감성 분석 결과")
    if not raw_news_df.empty:
        news_weekly = st.session_state.get('weekly_dfs', {}).get('news', pd.DataFrame())
        if not news_weekly.empty:
            fig = px.line(news_weekly, y='Sentiment', title="주별 평균 뉴스 감성 점수", labels={'Sentiment': '감성 점수'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("수집된 뉴스 기사 목록 (최신순)")
        st.dataframe(raw_news_df.sort_values(by='날짜', ascending=False))
    else:
        st.info("사이드바에서 '최신 뉴스 분석하기' 버튼을 눌러주세요.")

with tab4:
    st.header("상관관계 분석")
    weekly_dfs = st.session_state.get('weekly_dfs', {})
    if weekly_dfs:
        dfs_to_concat = [df for df in weekly_dfs.values() if not df.empty]
        if dfs_to_concat:
            final_df = reduce(lambda left, right: pd.merge(left, right, on='날짜', how='outer'), dfs_to_concat)
            final_df = final_df.interpolate(method='linear', limit_direction='both').dropna(how='all')
            st.session_state['final_df'] = final_df
            
            st.subheader("통합 데이터 시각화")
            df_long = final_df.reset_index().melt(id_vars='날짜', var_name='데이터 종류', value_name='값')
            fig = px.line(df_long, x='날짜', y='값', color='데이터 종류', title="통합 데이터 시계열 추이")
            st.plotly_chart(fig, use_container_width=True)

            if len(final_df.columns) > 1:
                st.markdown("---")
                st.subheader("상관관계 히트맵")
                corr_matrix = final_df.corr(numeric_only=True)
                fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1])
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("분석할 주별 데이터가 없습니다.")
    else:
        st.warning("2단계에서 데이터가 처리되지 않았습니다.")

with tab5:
    st.header("시계열 분해 및 예측 (by Prophet)")
    final_df = st.session_state.get('final_df', pd.DataFrame())

    if not final_df.empty:
        forecast_col = st.selectbox("예측 대상 변수 선택", final_df.columns, key="forecast_select")

        # 예측 실행 버튼 추가
        if st.button("📈 선택한 변수로 예측 실행하기"):
            ts_data = final_df[[forecast_col]].dropna()
            if len(ts_data) < 24:
                st.warning(f"시계열 분석을 위해 최소 24주 이상의 데이터가 필요합니다. 현재 데이터: {len(ts_data)}주")
            else:
                with st.spinner(f"'{forecast_col}'에 대한 예측 모델을 학습 중입니다..."):
                    # --- 시계열 분해 ---
                    st.subheader(f"'{forecast_col}' 시계열 분해")
                    period = 52 if len(ts_data) >= 104 else max(4, int(len(ts_data) / 2))
                    decomposition = seasonal_decompose(ts_data[forecast_col], model='additive', period=period)
                    
                    fig_decompose = go.Figure()
                    fig_decompose.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'))
                    fig_decompose.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                    fig_decompose.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                    # 결과를 세션 상태에 저장
                    st.session_state['fig_decompose'] = fig_decompose

                    # --- Prophet 예측 ---
                    st.subheader(f"'{forecast_col}' 미래 12주 예측")
                    prophet_df = ts_data.reset_index().rename(columns={'날짜': 'ds', forecast_col: 'y'})
                    
                    m = Prophet()
                    m.fit(prophet_df)
                    future = m.make_future_dataframe(periods=12, freq='W')
                    forecast = m.predict(future)

                    fig_forecast = plot_plotly(m, forecast)
                    fig_forecast.update_layout(title=f"'{forecast_col}' 미래 예측", xaxis_title='날짜', yaxis_title='예측값')
                    
                    # 결과를 세션 상태에 저장
                    st.session_state['fig_forecast'] = fig_forecast
                    st.session_state['forecast_data'] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
        
        # --- 저장된 결과 표시 ---
        # 버튼을 누른 후, 다른 위젯과 상호작용해도 결과가 사라지지 않도록 함
        if 'fig_decompose' in st.session_state:
            st.subheader("시계열 분해 결과")
            st.plotly_chart(st.session_state['fig_decompose'], use_container_width=True)

        if 'fig_forecast' in st.session_state and 'forecast_data' in st.session_state:
            st.subheader("미래 예측 결과")
            st.plotly_chart(st.session_state['fig_forecast'], use_container_width=True)
            st.write("#### 예측 데이터 테이블")
            st.dataframe(st.session_state['forecast_data'])

    else:
        st.info("4번 탭에서 데이터가 통합되어야 예측을 수행할 수 있습니다.")
