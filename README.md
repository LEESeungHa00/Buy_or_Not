📈 수입량 및 가격 예측 AI 대시보드
다양한 외부 데이터(뉴스, 검색 트렌드, 공공 데이터 등)를 종합하여 특정 품목의 미래 수입량과 가격을 예측하고, 시장 인사이트를 도출하는 머신러닝 기반의 웹 대시보드입니다.

✨ 주요 기능
통합 데이터 분석: 수입 데이터, 국내 도소매가, 온라인 검색량, 뉴스 데이터를 종합하여 시계열 트렌드를 시각화합니다.

미래 예측: XGBoost, LightGBM 등 머신러닝 모델을 활용하여 미래의 수입량과 수입 가격을 예측합니다.

심층 분석:

품목 분석: 특정 품목의 주요 지표(단가, 수요, 감성 지수) 간의 상관관계를 분석합니다.

수입사 분석: 주요 수입사들의 품목별 점유율과 가격 경쟁력을 비교 분석합니다.

자동화된 파이프라인: 데이터 수집, 전처리, 모델링, 시각화까지의 과정을 파이썬 코드로 자동화합니다.

🏗️ 시스템 아키텍처
이 프로젝트는 데이터 수집, 전처리, 모델링, 시각화의 4단계 파이프라인으로 구성되어 있습니다. 모든 데이터는 SQLite 데이터베이스에 통합 관리하여 효율성을 높였습니다.



🛠️ 기술 스택
언어: Python 3.9+

데이터 처리/분석: Pandas, Numpy

머신러닝: Scikit-learn, XGBoost, LightGBM

데이터베이스: SQLite3

웹 프레임워크/시각화: Streamlit, Plotly Express

데이터 수집 : Requests, BeautifulSoup4, Pytrends

🚀 시작하기
1. 프로젝트 복제
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

2. 가상환경 생성 및 활성화
# Windows
python -m venv venv
venv\Scripts\activate

# macOS & Linux
python3 -m venv venv
source venv/bin/activate

3. 필요 라이브러리 설치
pip install -r requirements.txt

4. 폴더 구조 설정
프로젝트 루트 디렉토리에 아래와 같이 폴더를 생성해주세요.

/your-repository-name
|-- /data/              # DB 파일, 초기 CSV 데이터 등 위치
|-- /models/            # 학습된 모델 파일(.json) 저장
|-- /config/            # 키워드 매핑 파일(keywords.json) 등 위치
|-- /src/               # 파이썬 소스코드
|-- app.py              # Streamlit 실행 파일
|-- ...

5. Streamlit 앱 실행
streamlit run app.py

브라우저에서 http://localhost:8501 주소로 접속하여 대시보드를 확인합니다.

📁 파일 구조
.
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
└── src
    ├── __init__.py
    ├── data_processing.py
    └── modeling.py
