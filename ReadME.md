📈 AI 기반 수입 데이터 분석 및 예측 대시보드
<div align="center">
<img src="https://www.google.com/search?q=https://placehold.co/800x250/0078D4/FFFFFF%3Ftext%3DAI%2BTrade%2BData%2BDashboard" alt="Dashboard Banner">
</div>

<br>

수입 데이터, 국내 도매가, 뉴스 기사, 검색 트렌드 등 파편화된 데이터를 Google BigQuery에 통합하고, 상관관계 분석과 AI 예측 모델을 통해 미래 시장을 예측하는 인사이트 대시보드입니다.

🎯 The Problem
데이터 기반의 정확한 수입 전략을 수립하는 데에는 다음과 같은 여러 어려움이 있었습니다.

🗂️ 파편화된 데이터: 수입 실적, 국내 시장 가격, 뉴스, 소비자 관심도 등 분석에 필요한 데이터가 여러 곳에 흩어져 있어 통합적인 분석이 어려웠습니다.

🤔 직감에 의존한 의사결정: 명확한 데이터 근거 없이 과거의 경험이나 직감에 의존하여 다음 분기의 수입량과 가격 전략을 수립했습니다.

📉 숨겨진 관계 파악의 어려움: 온라인 검색량이나 특정 뉴스 이슈가 실제 수입량과 단가에 얼마나, 그리고 언제 영향을 미치는지 파악하기 힘들었습니다.

🔮 미래 예측의 부재: 미래의 수요와 가격 변동을 예측할 수 없어, 재고 관리와 가격 협상에서 불리한 위치에 놓이는 경우가 많았습니다.

💡 The Solution
이러한 문제들을 해결하기 위해, 다양한 데이터 소스를 실시간으로 연동하고 AI 분석 기능을 탑재한 통합 분석 대시보드를 구축했습니다.

⚙️ 빅데이터 파이프라인 구축: 수십만 건의 수출입 데이터를 Google BigQuery에 저장하여, 대용량 데이터도 빠르고 안정적으로 처리하는 기반을 마련했습니다.

🤖 자동화된 외부 데이터 연동: KAMIS(농산물 가격), 네이버 데이터랩(검색/쇼핑 트렌드) 등 다양한 외부 데이터를 API로 자동 연동하고, 그 결과를 BigQuery에 캐싱하여 반복 조회 시 속도를 극대화했습니다.

🧠 AI 기반 뉴스 감성 분석: 금융/경제 뉴스에 특화된 AI 모델(KR-FinBERT)을 활용해 실시간으로 뉴스 여론을 분석하고, 나아가 AI가 왜 그런 판단을 내렸는지 근거가 된 핵심 단어까지 시각화(XAI)합니다.

🔗 심층 상관관계 분석: 히트맵과 시차(Lag) 분석을 통해, 선물 가격이나 뉴스 감성 점수와 같은 선행 지표가 미래의 수입량/단가에 미치는 숨겨진 선후행 관계를 명확하게 진단합니다.

🔭 AI 미래 예측 모델: 수집된 모든 데이터를 바탕으로 **시계열을 분해(Decomposition)**하여 데이터의 구조를 파악하고, Prophet 예측 모델을 통해 미래의 수입량, 단가 등을 예측하여 전략 수립을 지원합니다.

🛠️ Tech Stack
<div align="center">
<br>
<strong>Core & Visualization</strong><br>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Python-3776AB%3Fstyle%3Dfor-the-badge%26logo%3Dpython%26logoColor%3Dwhite" />
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Streamlit-FF4B4B%3Fstyle%3Dfor-the-badge%26logo%3Dstreamlit%26logoColor%3Dwhite" />
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Pandas-150458%3Fstyle%3Dfor-the-badge%26logo%3Dpandas%26logoColor%3Dwhite" />
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Plotly-3F4F75%3Fstyle%3Dfor-the-badge%26logo%3Dplotly%26logoColor%3Dwhite" />
<br><br>
<strong>Data Storage & Fetching</strong><br>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Google%2520BigQuery-4285F4%3Fstyle%3Dfor-the-badge%26logo%3Dgoogle-bigquery%26logoColor%3Dwhite" />
<img src="https://www.google.com/search?q=https://img.shields.io/badge/pandas_gbq-D72D49%3Fstyle%3Dfor-the-badge%26logo%3Dgoogle-cloud%26logoColor%3Dwhite" />
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Naver%2520API-03C75A%3Fstyle%3Dfor-the-badge%26logo%3Dnaver%26logoColor%3Dwhite" />
<br><br>
<strong>AI & Analysis</strong><br>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Transformers-FFD21E%3Fstyle%3Dfor-the-badge%26logo%3Dhugging-face%26logoColor%3Dblack" />
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Prophet-0078D4%3Fstyle%3Dfor-the-badge%26logo%3Dmeta%26logoColor%3Dwhite" />
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Scikit--learn-F7931E%3Fstyle%3Dfor-the-badge%26logo%3Dscikit-learn%26logoColor%3Dwhite" />
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Statsmodels-1A568C%3Fstyle%3Dfor-the-badge%26logo%3Dpython%26logoColor%3Dwhite" />
<br>
</div>

🚀 Getting Started
Prerequisites
Python 3.9+

Google Cloud Platform 계정 및 secrets.toml 설정 (아래 참조)

Naver, KAMIS 등 외부 API 키

Installation & Setup
저장소 복제 (Clone the repository):

git clone [저장소 URL]
cd [프로젝트 폴더]

필요한 라이브러리 설치:

pip install -r requirements.txt

Google Cloud 인증 설정:
프로젝트 폴더 내에 .streamlit/secrets.toml 파일을 생성하고, BigQuery 연동 설정 안내서에 따라 서비스 계정 키 내용을 붙여넣습니다.

Streamlit 앱 실행:

streamlit run data_explorer_app.py
