# -*- coding: utf-8 -*-
import streamlit as st
import base64
import os
import json
import pandas as pd
from typing import List, Dict

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from textwrap import fill

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 🔹 상수 정의 (모델 및 인덱스)
# ==========================================
EMBEDDING_MODEL = "jeffh/intfloat-multilingual-e5-large-instruct:q8_0"
CHAT_MODEL = "llama3.1:8b"

FAISS_INDEX_DRUG = "faiss_drug_index"
FAISS_INDEX_DISEASE = "faiss_disease_index"
FAISS_INDEX_PROCEDURE = "faiss_procedure_index"

# 관련 없는 문서 검색을 막기 위한 유사도 임계값 (L2 거리 기준, 낮을수록 유사)
RELEVANCE_THRESHOLD = 1.0

# ==========================================
# 🔹 백엔드: 임베딩 및 벡터스토어
# ==========================================

class InstructEmbeddings(OllamaEmbeddings):
    """ 'Represent this sentence for retrieval:' 구문을 추가하는 커스텀 임베딩 """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        instructed_texts = [f"Represent this sentence for retrieval: {text}" for text in texts]
        print(f"--- 📄 {len(instructed_texts)}개 문서 임베딩 중... ---")
        return super().embed_documents(instructed_texts)
    def embed_query(self, text: str) -> List[float]:
        instructed_text = f"Represent this sentence for retrieval: {text}"
        print(f"--- ❓ 쿼리 임베딩 중... ---")
        return super().embed_documents([instructed_text])[0]

@st.cache_resource
def get_embeddings():
    try:
        embeddings = InstructEmbeddings(model=EMBEDDING_MODEL)
        return embeddings
    except Exception as e:
        st.error(f"❌ 임베딩 모델({EMBEDDING_MODEL}) 초기화 중 오류: {e}"); st.stop()

def load_or_create_faiss_index(index_path: str, documents: List[Document], embeddings: InstructEmbeddings):
    if os.path.exists(index_path):
        try:
            print(f"--- 🚀 로컬 인덱스({index_path}) 로딩 시도... ---")
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("--- ✅ 로컬 인덱스 로드 완료 ---")
            return vector_store
        except Exception as e:
            st.warning(f"⚠️ 로컬 인덱스({index_path}) 로드 실패: {e}. 새 인덱스를 생성합니다.")
    try:
        print(f"--- ⏳ 새 벡터스토어 생성 중 ({index_path}) ---")
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        print(f"--- ✅ 새 벡터스토어 생성 및 {index_path}에 저장 완료 ---")
        return vector_store
    except Exception as e:
        st.error(f"❌ 데이터 임베딩/벡터스토어 생성 중 오류 ({index_path}): {e}"); st.stop()

@st.cache_resource
def load_drug_data(_embeddings):
    print("--- 🩺 [1/3] 약물 데이터 로딩 시작 ---")
    try:
        with open("drug_list.json", "r", encoding="utf-8") as f:
            j = json.load(f); all_items = j['body']['items'] if 'body' in j else (j if isinstance(j, list) else [])
    except FileNotFoundError: st.error("❌ drug_list.json 파일을 찾을 수 없습니다."); st.stop()
    all_documents = []
    for item in all_items:
        base_metadata = { "제품명": item.get('itemName', 'N/A'), "업체명": item.get('entpName', 'N/A'), "source": "drug_list.json" }
        sections = {"효능": item.get('efcyQesitm', 'N/A'), "사용법": item.get('useMethodQesitm', 'N/A'), "주의사항경고": item.get('atpnWarnQesitm', 'N/A'), "주의사항": item.get('atpnQesitm', 'N/A'), "상호작용": item.get('intrcQesitm', 'N/A'), "부작용": item.get('seQesitm', 'N/A'), "보관법": item.get('depositMethodQesitm', 'N/A')}
        for sec_name, sec_content in sections.items():
            if sec_content not in ('N/A', '', None, ' '):
                all_documents.append(Document(page_content=f"{sec_name}: {sec_content}", metadata={**base_metadata, "section": sec_name}))
    if not all_documents: st.error("❌ drug_list.json에서 유효한 문서를 찾지 못했습니다."); st.stop()
    return load_or_create_faiss_index(FAISS_INDEX_DRUG, all_documents, _embeddings), len(all_items)

@st.cache_resource
def load_disease_data(_embeddings):
    print("--- 🩺 [2/3] 질병/증상 데이터 로딩 시작 ---")
    try: df = pd.read_csv("textbook.csv")
    except FileNotFoundError: st.error("❌ textbook.csv 파일을 찾을 수 없습니다."); st.stop()
    if "content" not in df.columns: st.error("❌ textbook.csv 파일에 'content' 컬럼이 없습니다."); st.stop()
    all_documents = [Document(page_content=row["content"], metadata={"source": "textbook.csv"}) for _, row in df.iterrows() if pd.notna(row["content"]) and row["content"].strip()]
    if not all_documents: st.error("❌ textbook.csv에서 유효한 문서를 찾지 못했습니다."); st.stop()
    return load_or_create_faiss_index(FAISS_INDEX_DISEASE, all_documents, _embeddings), len(all_documents)

@st.cache_resource
def load_procedure_data(_embeddings):
    print("--- 🩺 [3/3] 수술/시술 데이터 로딩 시작 ---")
    try: df = pd.read_csv("etc.csv")
    except FileNotFoundError: st.error("❌ etc.csv 파일을 찾을 수 없습니다."); st.stop()
    if "content" not in df.columns: st.error("❌ etc.csv 파일에 'content' 컬럼이 없습니다."); st.stop()
    all_documents = [Document(page_content=row["content"], metadata={"source": "etc.csv"}) for _, row in df.iterrows() if pd.notna(row["content"]) and row["content"].strip()]
    if not all_documents: st.error("❌ etc.csv에서 유효한 문서를 찾지 못했습니다."); st.stop()
    return load_or_create_faiss_index(FAISS_INDEX_PROCEDURE, all_documents, _embeddings), len(all_documents)


# ==========================================
# 🔹 백엔드: 데이터 로딩 실행
# ==========================================
embeddings = get_embeddings()
with st.spinner(f"🩺 의료 데이터베이스 준비 중... (Embedding: {EMBEDDING_MODEL})"):
    vector_store_drug, num_drugs = load_drug_data(embeddings)
    vector_store_disease, num_diseases = load_disease_data(embeddings)
    vector_store_procedure, num_procedures = load_procedure_data(embeddings)
    if not all([vector_store_drug, vector_store_disease, vector_store_procedure]):
        st.error("데이터 로딩에 실패했습니다. 앱을 다시 시작하세요."); st.stop()

# ==========================================
# 🔹 백엔드: 4-Chain 정의
# ==========================================

# --- 1. 질문 재작성기 (Contextualizer) ---
try:
    contextualizer_model = ChatOllama(model=CHAT_MODEL, temperature=0.0)
    contextualizer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 '대화 기록'을 바탕으로 '새 질문'을 독립적으로 검색 가능한 '완전한 질문'으로 재작성하는 AI입니다.
- '새 질문'이 '그거', '저거', '어때' 등 맥락에 의존한다면, '대화 기록'을 참고하여 완전한 질문으로 만드세요.
- '새 질문'이 이미 완전하다면, 그대로 반환하세요.
- 오직 재작성된 질문 "한 문장"만 대답하세요.

[예시 1]
대화 기록: 
  Human: 두통약 알려줘
  AI: 타이레놀을 추천합니다.
새 질문: 그거 부작용은 뭐야?
당신 DML 답변: 타이레놀의 부작용은 무엇인가요?

[예시 2]
대화 기록: (없음)
새 질문: 감염성 대장염이 뭐야?
당신 DML 답변: 감염성 대장염이 뭐야?
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "새 질문: {question}")
    ])
    contextualizer_chain = contextualizer_prompt | contextualizer_model | StrOutputParser()
except Exception as e:
    st.error(f"❌ 질문 재작성기 모델({CHAT_MODEL}) 로드 중 오류: {e}"); st.stop()

# --- 2. 라우터(Router) ---
try:
    router_model = ChatOllama(model=CHAT_MODEL, temperature=0)
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 사용자의 질문을 '약물', '질병', '수술' 세 가지 카테고리 중 하나로 분류하는 AI입니다.
- '약'에 대해 물으면 'drug'
- '병'이나 '증상'에 대해 물으면 'disease'
- '수술'이나 '시술'에 대해 물으면 'procedure'
- 어느 것에도 해당하지 않으면 'general'
이라고, 반드시 한 단어로만 대답하세요.
"""),
        ("user", "{question}")
    ])
    router_chain = router_prompt | router_model | StrOutputParser()
except Exception as e:
    st.error(f"❌ 라우터 모델({CHAT_MODEL}) 로드 중 오류: {e}"); st.stop()

# --- 3. 요약기(Summarizer) ---
try:
    summarizer_model = ChatOllama(model=CHAT_MODEL)
    # [수정됨] '기계' 대신 'AI 의료 조수'로, '전문가 톤' 대신 '친절한 톤'으로 수정
    summarizer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 '참고 자료'와 '대화 기록'을 바탕으로 사용자의 '현재 질문'에 친절하게 답변하는 AI 의료 조수입니다.

[중요 원칙]
1.  **자료 기반 답변 (Grounding):** 당신은 **오직** 제공된 '참고 자료'의 내용을 **요약**하거나 **인용**해야 합니다. 자료에 없는 내용은 절대 지어내지 마세요.
2.  **친절한 톤:** 전문가의 입장에서, 하지만 친절하고 이해하기 쉬운 말투로 답변하세요.
3.  **자료 없음 처리:** '참고 자료'가 "일치하는 항목을 찾지 못했습니다." 또는 "자료가 없습니다." 또는 "관련성이 높은 항목을 찾지 못했습니다."라고 반환되면, "죄송합니다. 요청하신 내용과 일치하는 정보를 찾지 못했습니다."라고 답변해야 합니다.
4.  사용자의 '현재 질문'에 대한 답변 형식으로 요약하세요. '대화 기록'을 참고하여 맥락에 맞는 답변을 하세요.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "참고 자료: \n{context}\n\n현재 질문: {question}")
    ])
    summarizer_chain = summarizer_prompt | summarizer_model | StrOutputParser()
except Exception as e:
    st.error(f"❌ 요약기 모델({CHAT_MODEL}) 로드 중 오류: {e}"); st.stop()

# --- 4. 추천 질문 생성기(Recommender) ---
try:
    recommender_model = ChatOllama(model=CHAT_MODEL, temperature=0.5) 
    recommender_prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 사용자에게 도움이 되는 '후속 질문'을 제안하는 AI 조수입니다.
제공된 'AI 답변'을 바탕으로, 사용자가 다음에 궁금해할 만한 3가지의 짧고 관련성 높은 질문을 생성해주세요.

[중요 규칙]
- 'AI 답변'이 "죄송합니다", "찾지 못했습니다" 등 거절의 내용이라면, "정보 없음"이라고만 대답하세요.
- 각 질문은 '• '로 시작하고, 줄바꿈으로 구분합니다.
- 오직 3개의 질문만 생성하고, 다른 말은 절대 덧붙이지 마세요.

[좋은 예시]
• 관련 약물의 부작용은 무엇인가요?
• 이 질병을 예방하는 방법이 있나요?
• 수술 후 회복 기간은 얼마나 걸리나요?
"""),
        ("user", "AI 답변:\n{answer}")
    ])
    recommender_chain = recommender_prompt | recommender_model | StrOutputParser()
except Exception as e:
    st.error(f"❌ 추천 모델({CHAT_MODEL}) 로드 중 오류: {e}"); st.stop()


# ==========================================
# 🔹 백엔드: 검색 도구 (Retriever)
# ==========================================

# [!!!] 여기가 수정된 부분입니다.
def retrieve_drug_info(query: str, k: int):
    """[약사] '효능' 필터 [제거] + Score Threshold 적용."""
    
    # k값보다 여유있게 검색 (k*2 + 2)
    retrieved_docs_with_scores = vector_store_drug.similarity_search_with_score(query, k=(k*2 + 2)) 
    
    # 임계값(Threshold)을 통과한 문서만 1차로 필터링
    threshold_docs = [doc for doc, score in retrieved_docs_with_scores if score < RELEVANCE_THRESHOLD]
    
    # [수정] '효능' 섹션만 필터링하던 라인을 제거하고, 임계값을 통과한 모든 문서를 후보로 봅니다.
    # filtered_docs = [doc for doc in threshold_docs if doc.metadata.get("section") == "효능"]
    
    # 임계값을 통과한 문서들 중 최종 k개만 선택
    final_docs = threshold_docs[:k] 
    
    if not final_docs:
        # [수정] 실패 메시지를 더 일반적인 내용으로 수정
        return "검색된 약물 자료 중 관련성이 높은 항목을 찾지 못했습니다."
    
    formatted_docs = []
    for i, doc in enumerate(final_docs, 1):
        item_name = doc.metadata.get("제품명", "N/A")
        # [수정] 섹션 이름을 하드코딩('효능')하지 않고, 메타데이터에서 동적으로 가져옵니다.
        section_name = doc.metadata.get("section", "정보 없음") 
        formatted_docs.append(
            # [수정] 동적으로 가져온 section_name을 표시합니다.
            f"📘 [약물 자료 {i}] {item_name} (섹션: {section_name})\n{'-'*20}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_docs)

def retrieve_disease_info(query: str, k: int):
    """[의사] 질병/증상 검색 + Score Threshold 적용"""
    retrieved_docs_with_scores = vector_store_disease.similarity_search_with_score(query, k=k) # 사이드바 k값 적용
    filtered_docs = [doc for doc, score in retrieved_docs_with_scores if score < RELEVANCE_THRESHOLD]

    if not filtered_docs: 
        return "검색된 질병/증상 자료가 없습니다."
    
    formatted_docs = []
    for i, doc in enumerate(filtered_docs, 1):
        formatted_docs.append(
            f"📗 [질병/증상 자료 {i}]\n{'-'*20}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_docs)

def retrieve_procedure_info(query: str, k: int):
    """[외과의] 수술/시술 검색 + Score Threshold 적용"""
    retrieved_docs_with_scores = vector_store_procedure.similarity_search_with_score(query, k=k) # 사이드바 k값 적용
    filtered_docs = [doc for doc, score in retrieved_docs_with_scores if score < RELEVANCE_THRESHOLD]

    if not filtered_docs: 
        return "검색된 수술/시술 자료가 없습니다."
    
    formatted_docs = []
    for i, doc in enumerate(filtered_docs, 1):
        formatted_docs.append(
            f"📙 [수술/시술 자료 {i}]\n{'-'*20}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_docs)


# ==========================================
# 🔹 UI: 유틸 (이미지 → base64 변환)
# ==========================================
@st.cache_data
def img_to_base64(path):
    # 파일이 존재하는지 확인
    if not os.path.exists(path):
        st.warning(f"이미지 파일을 찾을 수 없습니다: {path}. 기본 헤더로 대체합니다.")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# blue_medi.jpg, pink_medi.jpg 파일이 스크립트와 동일한 위치에 있어야 함
blue64 = img_to_base64("blue_medi.jpg") 
pink64 = img_to_base64("pink_medi.jpg")

# ==========================================
# 🔹 UI: 페이지 설정
# ==========================================
st.set_page_config(page_title="💊 SafeMedi AI", layout="wide")

# ==========================================
# 🔹 UI: 전역 스타일 (CSS)
# ==========================================
st.markdown("""
<style>
:root{
  --bg:#f8fbfd;
  --card:#ffffff;
  --mint:#A8E6CF;
  --sky:#B8E4F0;
  --coral:#FF9AA2;
  --ink:#1f2937;
  --sub:#6b7280;
  --ring: 0 0 0 3px rgba(184,228,240,.35);
}

/* 전체 배경 */
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }

header[data-testid="stHeader"] { background: transparent; }
section.block-container{ padding-top: 1.2rem; max-width: 960px; }

h1,h2,h3 { font-weight: 700; color: var(--ink); }

/* 헤더 */
.safe-header{
  display:flex; align-items:center; justify-content:space-between;
  background:linear-gradient(135deg, var(--sky), var(--mint));
  color:#003049; padding:18px 24px; border-radius:18px;
  box-shadow: 0 10px 24px rgba(0,0,0,.05);
}
.safe-header .left{
  display:flex; align-items:center; gap:1rem;
}
.safe-header img{
  width:65px; height:auto;
}
.safe-header .title{ font-size:22px; font-weight:700; }
.safe-header .desc{ font-size:13px; opacity:.85; }

/* 채팅 영역 */
.chat-wrap{
  margin-top:14px; background: var(--card); border-radius:24px;
  padding:14px 14px 6px; box-shadow: 0 8px 24px rgba(0,0,0,.06);
  border:1px solid rgba(0,0,0,.04);
  min-height: 400px;
}

/* 말풍선 */
.bubble{
  max-width: 84%; padding:12px 14px; border-radius:16px;
  margin:8px 0; line-height:1.45; word-break:keep-all; color:var(--ink);
  box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
.bubble.user{
  margin-left:auto; background: var(--sky);
  border-top-right-radius:6px;
}
.bubble.ai{
  background: #fff; border:1px solid rgba(0,0,0,.05);
  border-top-left-radius:6px;
}

/* 사이드바 */
[data-testid="stSidebar"]{
  background:#ffffffcc; backdrop-filter: blur(6px);
  border-right:1px solid rgba(0,0,0,.06);
}
.sidebar-card{
  background:#fff; border:1px solid rgba(0,0,0,.05);
  border-radius:16px; padding:12px; margin-bottom:10px;
  box-shadow: 0 6px 16px rgba(0,0,0,.05);
}

/* 버튼 & 입력창 */
.stButton>button{
  background: var(--coral); color:white; border:0; padding:.6rem 1rem;
  border-radius:12px; font-weight:700;
  box-shadow: 0 6px 18px rgba(255,154,162,.35);
}
.stButton>button:hover{ filter:brightness(.97); }

.stTextInput>div>div>input{
  border-radius:12px !important; border:1px solid rgba(0,0,0,.08);
  box-shadow: var(--ring);
}

/* 푸터 */
footer.note{
  margin-top:10px; font-size:12px; color:#64748b;
  text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔹 UI: 헤더 (이미지 포함)
# ==========================================
# 이미지가 로드되었을 때만 헤더를 표시
if blue64 and pink64:
    st.markdown(f"""
    <div class="safe-header">
      <div class="left">
        <img src="data:image/jpeg;base64,{blue64}" alt="blue medi">
        <div>
          <div class="title">SafeMedi AI</div>
          <div class="desc">귀엽고 직관적인 약물 안전 상담 챗봇</div>
        </div>
      </div>
      <img src="data:image/jpeg;base64,{pink64}" alt="pink medi">
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("💊 SafeMedi AI")
    st.caption("귀엽고 직관적인 약물 안전 상담 챗봇")


# ==========================================
# 🔹 UI: 말풍선 함수
# ==========================================
def bubble(role:str, text:str):
    cls = "ai" if role=="ai" else "user"
    st.markdown(f'<div class="bubble {cls}">{text}</div>', unsafe_allow_html=True)

# ==========================================
# 🔹 UI: 사이드바
# ==========================================
with st.sidebar:
    # DB 로딩 상태 표시
    st.success(f"DB 준비 완료\n(약물: {num_drugs}개, 질병: {num_diseases}개, 수술/시술: {num_procedures}개)")
    st.markdown(f"<div style='font-size:12px; margin-top:-10px; margin-bottom:10px;'><b>LLM:</b> {CHAT_MODEL}</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card"><b>검색 옵션</b></div>', unsafe_allow_html=True)
    # k 값을 slider에서 받아옴
    k_slider = st.slider("문서 검색 개수 (k)", 3, 7, 4)
    only_preg = st.checkbox("임부금기 정보 우선 (미적용)") # TODO: 향후 로직 추가
    
    st.markdown('<div class="sidebar-card"><b>도움말</b><br/>약 이름이나 증상으로 질문해보세요 💬</div>', unsafe_allow_html=True)
    if st.button("대화 초기화"):
        st.session_state.pop("history", None)
        st.rerun()

# ==========================================
# 🔹 UI: 채팅 영역
# ==========================================
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []  # [(role, text)]

# 이전 대화 기록을 모두 표시
for role, txt in st.session_state.history:
    bubble(role, txt)

# 채팅 입력창
query = st.chat_input("예: 임신 중 타이레놀 복용해도 되나요?")

if query:
    # 1. 사용자 질문 표시 및 기록
    st.session_state.history.append(("user", query))
    bubble("user", query)
    
    original_question = query
    
    # 2. 대화 기록(Memory) 준비 (LangChain 형식)
    langchain_history = []
    # st.session_state.history 에는 (role, text) 튜플이 저장됨
    # [:-1]를 사용해 현재 입력된 질문은 제외하고 이전 기록만 참조
    for role, text in st.session_state.history[:-1]:
        if role == 'user':
            langchain_history.append(HumanMessage(content=text))
        else:
            langchain_history.append(AIMessage(content=text))

    # 3. [Chain 1] 질문 재작성 (Contextualizer)
    with st.spinner("질문 의도 이해 중... (재작성)"):
        try:
            rewritten_question = contextualizer_chain.invoke({
                "chat_history": langchain_history,
                "question": original_question
            }).strip()
            print(f"--- ❓ 원본 질문: {original_question} ---")
            print(f"--- 🔄 재작성된 질문: {rewritten_question} ---")
        except Exception as e:
            st.error(f"❌ 질문 재작성 오류: {e}"); st.stop()

    # 4. [Chain 2] 라우터 호출
    with st.spinner("질문 의도 분석 중... (라우팅)"):
        try:
            route_output = router_chain.invoke({"question": rewritten_question})
            route = route_output.strip().lower()
            print(f"--- 🧭 라우팅 결과: {route} ---")
        except Exception as e:
            st.error(f"❌ 라우터 실행 오류: {e}"); st.stop()
    
    # 5. [Tool] Python 로직으로 DB 검색
    response_text = ""
    context = None
    with st.spinner("전문가 DB 검색 중..."):
        try:
            # 사이드바의 k_slider 값을 사용
            if "drug" in route:
                context = retrieve_drug_info(rewritten_question, k_slider)
            elif "disease" in route:
                context = retrieve_disease_info(rewritten_question, k_slider)
            elif "procedure" in route:
                context = retrieve_procedure_info(rewritten_question, k_slider)
            else:
                context = None 
                response_text = "죄송합니다. 의료 정보(약물, 질병, 수술)와 관련된 질문만 답변할 수 있습니다."
        except Exception as e:
            st.error(f"❌ DB 검색 오류: {e}"); st.stop()

    # 6. [Chain 3] 요약기 호출 (Streaming)
    answer = ""
    with st.spinner("답변 생성 중…"):
        bubble_container = st.empty()
        streaming_answer = ""
        
        if context is not None:
            try:
                for chunk in summarizer_chain.stream({
                    "context": context, 
                    "question": rewritten_question,
                    "chat_history": langchain_history
                }):
                    streaming_answer += chunk
                    # 스트리밍되는 내용을 AI 버블로 실시간 표시
                    bubble_container.markdown(f'<div class="bubble ai">{streaming_answer}▌</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ 답변 생성 중 오류: {e}")
                streaming_answer = "답변 생성 중 오류가 발생했습니다."
        else:
            streaming_answer = response_text
        
        # 스트리밍 완료 후 최종 버블 표시
        bubble_container.markdown(f'<div class="bubble ai">{streaming_answer}</div>', unsafe_allow_html=True)
        answer = streaming_answer # 최종 답변 저장

    # 7. AI 답변을 세션 히스토리에 추가
    st.session_state.history.append(("ai", answer))

    # 8. [Chain 4] 추천 질문 생성
    is_successful_answer = (
        context is not None and 
        "오류가 발생했습니다" not in answer and
        "찾지 못했습니다" not in answer and
        "자료가 없습니다" not in answer
    )
    if is_successful_answer:
        with st.spinner("🔍 관련 질문 추천 중..."):
            try:
                recommendations_output = recommender_chain.invoke({"answer": answer})
                if "정보 없음" not in recommendations_Doutput:
                    # 추천 질문을 채팅창 내에 예쁘게 표시
                    st.markdown(f"""
                    <div style="margin-left: 20px; margin-top: -10px; margin-bottom: 10px; font-size: 14px; border-left: 3px solid var(--mint); padding-left: 10px;">
                        <b style="color:var(--ink);">💡 관련 추천 질문:</b><br>
                        {recommendations_output.replace('•', '• ').replace('\n', '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                print(f"--- ⚠️ 추천 질문 생성 실패: {e} ---")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<footer class="note">※ 본 서비스는 정보 제공용이며, 의료 전문의 상담을 대체하지 않습니다.</footer>', unsafe_allow_html=True)