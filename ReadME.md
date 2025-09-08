# 📈 AI 기반 수입 데이터 분석 및 예측 대시보드

<div align="center">
  <img src="https://placehold.co/800x250/0078D4/FFFFFF?text=AI+Trade+Data+Dashboard" alt="Dashboard Banner">
</div>

<br>

> **수입 데이터, 국내 도매가, 뉴스 기사, 검색 트렌드 등 파편화된 데이터를 Google BigQuery에 통합하고, 상관관계 분석과 AI 예측 모델을 통해 미래 시장을 예측하는 인사이트 대시보드입니다.**
> 데이터에 기반한 정확한 수입 전략 수립을 목표로 합니다.

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Badge"/>
  <img src="https://img.shields.io/badge/Google%20BigQuery-4285F4?style=for-the-badge&logo=google-bigquery&logoColor=white" alt="BigQuery Badge"/>
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=hugging-face&logoColor=black" alt="Transformers Badge"/>
  <img src="https://img.shields.io/badge/Prophet-0078D4?style=for-the-badge&logo=meta&logoColor=white" alt="Prophet Badge"/>
</p>

---

### ✨ 주요 기능

-   **⚙️ 빅데이터 파이프라인 구축:** 수십만 건의 수출입 데이터를 **Google BigQuery**에 저장하여, 대용량 데이터도 빠르고 안정적으로 처리하는 기반을 마련했습니다.
-   **🤖 자동화된 외부 데이터 연동:** **KAMIS(농산물 가격), 네이버 데이터랩(검색/쇼핑 트렌드)** 등 다양한 외부 데이터를 API로 자동 연동하고, 그 결과를 BigQuery에 캐싱하여 반복 조회 시 속도를 극대화했습니다.
-   **🧠 AI 기반 뉴스 감성 분석:** 금융/경제 뉴스에 특화된 AI 모델(**KR-FinBERT**)을 활용해 실시간으로 뉴스 여론을 분석하고, 나아가 AI가 왜 그런 판단을 내렸는지 근거가 된 핵심 단어까지 시각화(**XAI**)합니다.
-   **🔗 심층 상관관계 분석:** 히트맵과 **시차(Lag) 분석**을 통해, 선물 가격이나 뉴스 감성 점수와 같은 선행 지표가 미래의 수입량/단가에 미치는 숨겨진 선후행 관계를 명확하게 진단합니다.
-   **🔭 AI 미래 예측 모델:** 수집된 모든 데이터를 바탕으로 **시계열을 분해(Decomposition)**하여 데이터의 구조를 파악하고, **Prophet** 예측 모델을 통해 미래의 수입량, 단가 등을 예측하여 전략 수립을 지원합니다.

---

### 🛠️ 기술 스택 (Tech Stack)

<table>
  <tr>
    <td align="center"><strong>Category</strong></td>
    <td align="center"><strong>Skills</strong></td>
  </tr>
  <tr>
    <td><strong>Core & Visualization</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge"/>
      <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Badge"/>
      <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas Badge"/>
      <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly Badge"/>
    </td>
  </tr>
  <tr>
    <td><strong>Data Storage & Fetching</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Google%20BigQuery-4285F4?style=for-the-badge&logo=google-bigquery&logoColor=white" alt="BigQuery Badge"/>
      <img src="https://img.shields.io/badge/pandas_gbq-D72D49?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Pandas GBQ Badge"/>
      <img src="https://img.shields.io/badge/Naver%20API-03C75A?style=for-the-badge&logo=naver&logoColor=white" alt="Naver API Badge"/>
    </td>
  </tr>
  <tr>
    <td><strong>AI & Analysis</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=hugging-face&logoColor=black" alt="Transformers Badge"/>
      <img src="https://img.shields.io/badge/Prophet-0078D4?style=for-the-badge&logo=meta&logoColor=white" alt="Prophet Badge"/>
      <img src="https://img.shields.io/badge/Statsmodels-1A568C?style=for-the-badge&logo=python&logoColor=white" alt="Statsmodels Badge"/>
      <img src="https://img.shields.io/badge/Captum-C9E2F5?style=for-the-badge&logo=pytorch&logoColor=black" alt="Captum Badge"/>
    </td>
  </tr>
</table>

---

### 🚀 시작하기 (Getting Started)

#### **Prerequisites**
-   Python 3.9+
-   Google Cloud Platform 계정 및 `secrets.toml` 설정
-   Naver, KAMIS 등 외부 API 키

#### **Installation & Setup**
1.  **저장소 복제 (Clone the repository):**
    ```bash
    git clone [저장소 URL]
    cd [프로젝트 폴더]
    ```

2.  **필요한 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Google Cloud 인증 설정:**
    프로젝트 루트에 `.streamlit/secrets.toml` 파일을 생성하고, [BigQuery 연동 설정 안내서](<링크 삽입>)에 따라 서비스 계정 키 내용을 붙여넣습니다.

4.  **Streamlit 앱 실행:**
    ```bash
    streamlit run data_explorer_app.py
    ```
