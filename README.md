<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" width="120" alt="pill icon">
</p>

<h1 align="center">💊 의료 정보 상담챗봇 프로젝트</h1>

<p align="center">
  <strong>당신의 건강한 하루를 지키는 작은 AI 에이전트 🩺</strong><br><br>
</p>


<div align=center> 
  
## 🫂 프로젝트 소개  

> “이 약, 어떻게 먹어야 하지?”  
> “두통약이랑 감기약 같이 먹어도 될까?”



 우리의 **의료 상담 챗봇**은<br>
 식품의약품안전처 공공데이터를 기반으로,<br>
 증상·약품명·효능·사용법·부작용 등 복잡한 정보를 쉽고 빠르게 알려주는<br>
 **RAG 기반 AI 4-Chain 의료 챗봇** 이에요. 👩‍⚕️<br><br>
 
<br><br>

##  SKILL STACK


<div align=center> 
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" height="60"/>
<img src="https://img.shields.io/badge/FAISS-111F68?style=for-the-badge&logo=meta&logoColor=white" height="60"/>
<img src="https://img.shields.io/badge/LangChain-3C4F76?style=for-the-badge&logo=https://logo.svgcdn.com/simple-icons/langchain-dark.png&logoColor=white" height="60"/>
<img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" height="60"/>
</br>
<img src="https://img.shields.io/badge/Streamlit-FF4F00?style=for-the-badge&logo=streamlit&logoColor=white" height="60"/>
<img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" height="60"/>
<img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" height="60"/>

</div><br><br>

##  CREW

| 이름 | 역할 | 주요 담당 |
|------|------|-----------|
| **채서린** | 🩺 병원장 | 프로젝트 총괄, RAG 파이프라인 설계 및 챗봇 구현 |
| **천현진** | 💾 데이터·백엔드 | 약품 리스트 파싱, 벡터DB 구성, API 설계 |
| **박현욱** | 💽 데이터·백엔드 | 데이터 전처리, DB 구축, API 구성 |
| **황혜윤** | 💻 프론트엔드/UI | 챗봇 인터페이스 디자인 및 구현 (CLI·웹) |
| **정지아** | 🩹 문서화/테스트 | 프로젝트 문서화, QA, 발표 자료 제작 |
| **염한결** | 🚑 Presenter | QA, 발표 자료 제작 |

<br><br>



---

<div align=left>

## 🖥️ 주요 기능 
- **약품명/효능/사용법/부작용 질의응답**: 다양한 약품 정보를 자동으로 안내<br>
- **유사도 기반 검색 및 요약**: 사용자의 애매한 질문을 '검색용 질문'으로 재작성해 가장 관련성이 높은 답변 제공<br>
- **한국어 지원 인터페이스**: CLI 또는 웹 기반으로 누구나 이용<br>
- **정확한 데이터 근거 답변**: 공식 데이터만 기반, 무책임 생성 응답 방지<br><br>


<br>

## 챗봇 질문 예시

```
1. "아랫배가 아픈데 어떤 병이 있을까?"
2. "타이레놀의 부작용 알려줘"
3. "소화불량에 좋은 약 추천해줘"
4. "임산부가 먹으면 안되는 약은?"
5. "생리통이 너무 심해, 잘 듣는 진통제 알려줘"
6. "위장약의 사용법은?"
7. "아스피린 복용 시 주의사항은?"
```

<br><br>

---

## 🛠️ 직접 설치 및 실행 방법

### 사전 요구사항

- Python 3.8 이상 (권장: 3.10+)
- 최소 8GB RAM 및 5GB 디스크 공간
- Ollama 설치 및 모델 다운로드 (로컬 LLM 필수)
```
#Ollama 공식 사이트에서 설치 (https://ollama.com/download)
ollama --version
#모델 다운로드
ollama pull jeffh/intfloat-multilingual-e5-large-instruct:q8_0
ollama pull llama3.1:8b
```



- 프로젝트 클론 및 의존성 설치
```
git clone https://github.com/your-repo/drug-info-chatbot.git
cd drug-info-chatbot

python -m venv venv
Windows

venv\Scripts\activate
macOS/Linux

source venv/bin/activate

pip install -r requirements.txt
```

- 데이터 파일 준비
```
프로젝트 루트에 아래 파일이 필요합니다:
- drugchat.py
- requirements.txt
- drug_list.json
- textbook.csv
- etc.csv
```

- 챗봇 실행

```
streamlit run drugchat.py
#브라우저에서 http://localhost:8501 접속
#최초 실행 시 FAISS 인덱스 생성에 5~10분 소요  
#이후부터 즉시 시작됨
```
<br><br>


## 문제 해결 (Troubleshooting)

<details>
<summary>Ollama 모델 오류</summary>
  

ollama serve
ollama pull llama3.1:8b

text
</details>

<details>
<summary>faiss-cpu 설치 오류 (Apple Silicon)</summary>
  

pip uninstall faiss-cpu
pip install faiss-cpu --no-cache-dir

text
</details>

<details>
<summary>메모리 부족 오류</summary>

drugchat.py에서 k 값을 조정하세요.

retrieved_docs_with_scores = vector_store_drug.similarity_search_with_score(query, k=5)

text
</details>

<br><br>

---

## 🔧 프롬프트

- [핵심] **4-Chain 구조**: Query Rewriter -> Router -> Summarizer -> Recommender
- [핵심] **질문 재작성기(Contextualizer)** 를 추가하여 대화 맥락(Memory)을 '검색'에 반영
- [핵심] **similarity_search_with_score와 THRESHOLD**를 적용하여 관련 없는 문서(GIGO) 검색 차단<br><br>

1️⃣
```
당신은 '대화 기록'을 바탕으로 '새 질문'을 독립적으로 검색 가능한 '완전한 질문'으로 재작성하는 AI입니다.
  '새 질문'이 '그거', '저거', '어때' 등 맥락에 의존한다면, '대화 기록'을 참고하여 완전한 질문으로 만드세요.
  '새 질문'이 이미 완전하다면, 그대로 반환하세요.
  오직 재작성된 질문 '한 문장'만 대답하세요.
```

2️⃣
```
당신은 사용자의 질문을 '약물', '질병', '수술' 세 가지 카테고리 중 하나로 분류하는 AI입니다.
  - '약'에 대해 물으면 'drug'
  - '병'이나 '증상'에 대해 물으면 'disease'
  - '수술'이나 '시술'에 대해 물으면 'procedure'
  - 어느 것에도 해당하지 않으면 'general'
  이라고, 반드시 한 단어로만 대답하세요.
```
3️⃣
```
당신은 '참고 자료'와 '대화 기록'을 바탕으로 사용자의 '현재 질문'에 답변하는 기계입니다.
  [4대 원칙 (매우 중요)]"
  1.  **자료 기반 답변 (Grounding):** 당신은 **오직** 제공된 '참고 자료'의 내용을 **요약**하거나 **인용**해야 합니다.
  2.  **전문가 톤:** 불필요한 공감이나 대화체 문장을 **절대** 사용하지 마세요.
  3.  **자료 없음 처리:** '참고 자료'가 "일치하는 항목을 찾지 못했습니다."라고 반환되면, "죄송합니다. 요청하신 내용과 일치하는 정보를 찾지 못했습니다."라고 답변해야 합니다.
  4.  사용자의 '현재 질문'에 대한 답변 형식으로 요약하세요. '대화 기록'을 참고하여 맥락에 맞는 답변을 하세요.
```
4️⃣
```
당신은 사용자에게 도움이 되는 '후속 질문'을 제안하는 AI 조수입니다.
제공된 'AI 답변'을 바탕으로, 사용자가 다음에 궁금해할 만한 3가지의 짧고 관련성 높은 질문을 생성해주세요.
  [중요 규칙]
  - 'AI 답변'이 "죄송합니다", "찾지 못했습니다" 등 거절의 내용이라면, "정보 없음"이라고만 대답하세요.
  - 각 질문은 '• '로 시작하고, 줄바꿈으로 구분합니다.
  - 오직 3개의 질문만 생성하고, 다른 말은 절대 덧붙이지 마세요.
```

<br>


## 📋 데이터셋

[📚 **식품의약품안전처_의약품 개요정보(e약은요)**](https://www.data.go.kr/data/15075057/openapi.do)  

[📚 **필수의료 의학지식 데이터**](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&&srchDataRealmCode=REALM006&aihubDataSe=data&dataSetSn=71875)<br><br><br><br>



---

## 본 챗봇은 의료적인 전문성을 가지지 않습니다

전문적인 의학적 조언, 진단 또는 치료를 대체할 수 없으며,<br><br>
의학적 결정이나 건강 상태에 대한 우려가 있을 경우 반드시 전문 의료인과 상담하시기 바랍니다.

---
