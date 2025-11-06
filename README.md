<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" width="120" alt="pill icon">
</p>

<h1 align="center">💊 의료 정보 상담챗봇 프로젝트</h1>

<p align="center">
  <strong>당신의 건강한 하루를 지키는 작은 AI 에이전트 🩺</strong><br><br>
</p>


<div align=center> 
  
## 🫂 프로젝트 소개  
```
 “이 약, 어떻게 먹어야 하지?”  
 “두통약이랑 감기약 같이 먹어도 될까?”
```


 우리의 **의료 상담 챗봇**은<br>
 식품의약품안전처 공공데이터를 기반으로,<br>
 증상·약품명·효능·사용법·부작용 등 복잡한 정보를 쉽고 빠르게 알려주는<br>
 **RAG 기반(ChatGPT + 검색형)** 챗봇이에요. 👩‍⚕️<br><br>
 
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
<br><br>

## 🖥️ 주요 기능 
- **약품명/효능/사용법/부작용 질의응답**: 다양한 약품 정보를 자동으로 안내<br>
- **유사도 기반 검색 및 요약**: NLP 임베딩을 이용해 가장 관련성이 높은 답변 제공<br>
- **한국어 지원 인터페이스**: CLI 또는 웹 기반으로 누구나 이용<br>
- **정확한 데이터 근거 답변**: 공식 데이터만 기반, 무책임 생성 응답 방지<br><br>

<br><br>



## 직접 설치 및 실행 방법
webUI를 통해 모델을 사용할 수 있습니다.

1. Python 3.8 이상 설치
2. 필수 패키지 설치:

```
pip install -r requirements.txt
drugchat.py
```




## 프롬프트(뺄?말?)

```
    "당신은 약품 정보 데이터베이스에 접근할 수 있는 전문 약사 AI입니다. "
    "사용자의 질문에 대해 retrieve_drug_info 도구를 사용하여 정확한 약품 정보를 제공하세요. "
    "반드시 데이터베이스에 있는 정보만을 기반으로 답변하며, 없는 정보는 추측하지 마세요. "
    "약품명, 효능, 사용법, 부작용, 주의사항 등을 명확하게 설명하세요."
```


## 챗봇 활용 및 질문 예시

```
1. "활명수의 효능이 뭐야?"
2. "타이레놀의 부작용 알려줘"
3. "소화불량에 좋은 약 추천해줘"
4. "임산부가 먹으면 안되는 약은?"
5. "두통약 알려줘"
6. "위장약의 사용법은?"
7. "아스피린 복용 시 주의사항은?"
```


## 데이터셋

[📚 **식품의약품안전처_의약품 개요정보(e약은요)**](https://www.data.go.kr/data/15075057/openapi.do)  

[📚 **필수의료 의학지식 데이터**](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&&srchDataRealmCode=REALM006&aihubDataSe=data&dataSetSn=71875)<br><br><br><br>



---

## 본 프로젝트의 언어모델은 의료적인 전문성을 가지지 않습니다

실제 증상이 있을 경우 반드시 의사와 상담하시길 바랍니다.

---
