# -*- coding: utf-8 -*-
import streamlit as st
import base64
import os
import json
import pandas as pd
from typing import List, Dict
import html # HTML 이스케이프를 위해 추가

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

# [!!!] 수정: 임계값(THRESHOLD) 자체를 사용하지 않도록 로직을 변경합니다.
# RELEVANCE_THRESHOLD = 1.3 # <-- 이 변수를 더 이상 사용하지 않습니다.

# ==========================================
# 🔹 백엔드: 임베딩 및 벡터스토어
# ==========================================

# e5-instruct 모델의 올바른 사용법(passage:/query:)으로 변경
class InstructEmbeddings(OllamaEmbeddings):
    """ 'passage:'와 'query:'를 사용하는 커스텀 임베딩 """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # DB에 저장되는 문서는 'passage:'를 사용
        instructed_texts = [f"passage: {text}" for text in texts]
        print(f"--- 📄 {len(instructed_texts)}개 문서 임베딩 중... (Prefix: 'passage:') ---")
        return super().embed_documents(instructed_texts)
    
    def embed_query(self, text: str) -> List[float]:
        # 사용자의 질문은 'query:'를 사용
        instructed_text = f"query: {text}"
        print(f"--- ❓ 쿼리 임베딩 중... (Prefix: 'query:') ---")
        return super().embed_documents([instructed_text])[0]

@st.cache_resource
def get_embeddings():
    try:
        embeddings = InstructEmbeddings(model=EMBEDDING_MODEL)
        return embeddings
    except Exception as e:
        st.error(f"❌ 임베딩 모델({EMBEDDING_MODEL}) 초기화 중 오류: {e}"); st.stop()

def load_or_create_faiss_index(index_path: str, documents: List[Document], embeddings: InstructEmbeddings):
    # [!!!] 중요: 새 코드로 실행하기 전, 반드시 'faiss_...' 폴더 3개를 수동으로 삭제해야 합니다!
    if os.path.exists(index_path):
        try:
            print(f"--- 🚀 로컬 인덱스({index_path}) 로딩 시도... ---")
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("--- ✅ 로컬 인덱스 로드 완료 ---")
            return vector_store
        except Exception as e:
            st.warning(f"⚠️ 로컬 인덱스({index_path}) 로드 실패({e}). 새 인덱스를 생성합니다.")
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
@st.cache_resource
def get_contextualizer_chain():
    try:
        contextualizer_model = ChatOllama(model=CHAT_MODEL, temperature=0.0)
        contextualizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 '대화 기록'을 바탕으로 '새 질문'을 독립적으로 검색 가능한 '완전한 질문'으로 재작성하는 AI입니다.
- '새 질문'이 '그거', '저거', '어때' 등 맥락에 의존한다면, '대화 기록'을 참고하여 완전한 질문으로 만드세요.
- '새 질문'이 이미 완전하다면, 그대로 반환하세요.
- 오직 재작성된 질문 "한 문장"만 대답하세요.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "새 질문: {question}")
        ])
        return contextualizer_prompt | contextualizer_model | StrOutputParser()
    except Exception as e:
        st.error(f"❌ 질문 재작성기 모델({CHAT_MODEL}) 로드 중 오류: {e}"); st.stop()

# --- 2. 라우터(Router) ---
@st.cache_resource
def get_router_chain():
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
        return router_prompt | router_model | StrOutputParser()
    except Exception as e:
        st.error(f"❌ 라우터 모델({CHAT_MODEL}) 로드 중 오류: {e}"); st.stop()

# --- 3. 요약기(Summarizer) ---
@st.cache_resource
def get_summarizer_chain():
    try:
        summarizer_model = ChatOllama(model=CHAT_MODEL)
        summarizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 '참고 자료'와 '대화 기록'을 바탕으로 사용자의 '현재 질문'에 친절하게 답변하는 AI 의료 조수입니다.

[중요 원칙]
1.  **자료 기반 답변 (Grounding):** 당신은 **오직** 제공된 '참고 자료'의 내용을 **요약**하거나 **인용**해야 합니다. 자료에 없는 내용은 절대 지어내지 마세요.
2.  **친절한 톤:** 전문가의 입장에서, 하지만 친절하고 이해하기 쉬운 말투로 답변하세요.
3.  **자료 없음 처리:** '참고 자료'가 "검색된 약물 자료 중 관련성이 높은 항목을 찾지 못했습니다." 또는 "검색된 질병/증상 자료가 없습니다." 또는 "검색된 수술/시술 자료가 없습니다."라고 반환되면, "죄송합니다. 요청하신 내용과 일치하는 정보를 찾지 못했습니다."라고 답변해야 합니다.
4.  사용자의 '현재 질문'에 대한 답변 형식으로 요약하세요. '대화 기록'을 참고하여 맥락에 맞는 답변을 하세요.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "참고 자료: \n{context}\n\n현재 질문: {question}")
        ])
        return summarizer_prompt | summarizer_model | StrOutputParser()
    except Exception as e:
        st.error(f"❌ 요약기 모델({CHAT_MODEL}) 로드 중 오류: {e}"); st.stop()

# --- 4. 추천 질문 생성기(Recommender) ---
@st.cache_resource
def get_recommender_chain():
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
"""),
            ("user", "AI 답변:\n{answer}")
        ])
        return recommender_prompt | recommender_model | StrOutputParser()
    except Exception as e:
        st.error(f"❌ 추천 모델({CHAT_MODEL}) 로드 중 오류: {e}"); st.stop()

# 체인 로드
contextualizer_chain = get_contextualizer_chain()
router_chain = get_router_chain()
summarizer_chain = get_summarizer_chain()
recommender_chain = get_recommender_chain()


# ==========================================
# 🔹 백엔드: 검색 도구 (Retriever)
# ==========================================

def retrieve_drug_info(query: str, k: int):
    """[약사] '효능' 섹션을 우선순위로 재정렬."""
    
    # [!!!] 수정: 'similarity_search' -> 'max_marginal_relevance_search'로 변경
    # 'k=k*2' (e.g., 8)개의 다양한 문서를 먼저 찾음
    # 'fetch_k=50' (50개) 문서를 미리 보고 그 중에서 k*2개를 고름
    retrieved_docs = vector_store_drug.max_marginal_relevance_search(
        query, 
        k=k*2,    
        fetch_k=50 
    ) 
    
    if retrieved_docs:
        top_5_names = [doc.metadata.get("제품명", "N/A") for doc in retrieved_docs[:5]]
        print(f"--- 🔍 [Debug-MMR] Diverse docs for '{query}': {top_5_names} ---")
    
    # [!!!] 수정: MMR은 (doc, score)가 아닌 doc 리스트를 반환하므로, score(x[1]) 정렬 제거
    re_ranked_docs = sorted(
        retrieved_docs, # MMR로 찾은 다양한 문서 리스트
        key=lambda x: (0 if x.metadata.get("section") == "효능" else 1)
        # 1순위: 효능, 2순위: (원래 순서 - MMR이 이미 보장)
    )
    
    # 재정렬된 리스트에서 k개 선택
    final_docs = re_ranked_docs[:k]
    
    if not final_docs:
        return "검색된 약물 자료 중 관련성이 높은 항목을 찾지 못했습니다." 
    
    formatted_docs = []
    for i, doc in enumerate(final_docs, 1):
        item_name = doc.metadata.get("제품명", "N/A")
        section_name = doc.metadata.get("section", "정보 없음") 
        formatted_docs.append(
            f"📘 [약물 자료 {i}] {item_name} (섹션: {section_name})\n{'-'*20}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_docs)

def retrieve_disease_info(query: str, k: int):
    """[의사] 질병/증상 검색 (임계값 제거)"""
    # [!!!] 수정: 임계값(Threshold) 필터링 로직을 완전히 제거
    retrieved_docs = vector_store_disease.similarity_search(query, k=k)

    if not retrieved_docs: 
        return "검색된 질병/증상 자료가 없습니다." 
    
    formatted_docs = []
    for i, doc in enumerate(retrieved_docs, 1):
        formatted_docs.append(
            f"📗 [질병/증상 자료 {i}]\n{'-'*20}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_docs)

def retrieve_procedure_info(query: str, k: int):
    """[외과의] 수술/시술 검색 (임계값 제거)"""
    # [!!!] 수정: 임계값(Threshold) 필터링 로직을 완전히 제거
    retrieved_docs = vector_store_procedure.similarity_search(query, k=k)

    if not retrieved_docs: 
        return "검색된 수술/시술 자료가 없습니다." 
    
    formatted_docs = []
    for i, doc in enumerate(retrieved_docs, 1):
        formatted_docs.append(
            f"📙 [수술/시술 자료 {i}]\n{'-'*20}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_docs)


# ==========================================
# 🔹 UI: 유틸 (이미지 → base64 변환)
# ==========================================
@st.cache_data
def img_to_base64(path):
    # .png를 먼저 시도하고, 없으면 .jpg를 시도
    if not os.path.exists(path):
        jpg_path = path.replace(".png", ".jpg")
        if os.path.exists(jpg_path):
            path = jpg_path
        else:
            st.warning(f"이미지 파일을 찾을 수 없습니다: {path} 또는 {jpg_path}. 기본 헤더로 대체합니다.")
            return None
            
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

blue64 = img_to_base64("blue_medi.png") 
pink64 = img_to_base64("pink_medi.png")

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
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
header[data-testid="stHeader"] { background: transparent; }
section.block-container{ padding-top: 1.2rem; max-width: 960px; }
h1,h2,h3 { font-weight: 700; color: var(--ink); }
.safe-header{
  display:flex; align-items:center; justify-content:space-between;
  background:linear-gradient(135deg, var(--sky), var(--mint));
  color:#003049; padding:18px 24px; border-radius:18px;
  box-shadow: 0 10px 24px rgba(0,0,0,.05);
}
.safe-header .left{ display:flex; align-items:center; gap:1rem; }
.safe-header img{ width:65px; height:auto; }
.safe-header .title{ font-size:22px; font-weight:700; }
.safe-header .desc{ font-size:13px; opacity:.85; }
.chat-wrap{
  margin-top:14px; background: var(--card); border-radius:24px;
  padding:14px 14px 6px; box-shadow: 0 8px 24px rgba(0,0,0,.06);
  border:1px solid rgba(0,0,0,.04);
  /* min-height: 400px;  <-- 제거됨 */
}
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
[data-testid="stSidebar"]{
  background:#ffffffcc; backdrop-filter: blur(6px);
  border-right:1px solid rgba(0,0,0,.06);
}
.sidebar-card{
  background:#fff; border:1px solid rgba(0,0,0,.05);
  border-radius:16px; padding:12px; margin-bottom:10px;
  box-shadow: 0 6px 16px rgba(0,0,0,.05);
}
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
footer.note{
  margin-top:10px; font-size:12px; color:#64748b;
  text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔹 UI: 헤더 (이미지 포함)
# ==========================================
if blue64 and pink64:
    img_type = "png" if blue64.startswith("data:image/png") else "jpeg"
    st.markdown(f"""
    <div class="safe-header">
      <div class="left">
        <img src="data:image/{img_type};base64,{blue64}" alt="blue medi">
        <div>
          <div class="title">SafeMedi AI</div>
          <div class="desc">귀엽고 직관적인 약물 안전 상담 챗봇</div>
        </div>
      </div>
      <img src="data:image/{img_type};base64,{pink64}" alt="pink medi">
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
    
    if role == "user":
        text_to_render = html.escape(text)
    else:
        text_to_render = text # AI 답변(HTML)은 이스케이프하지 않음
        
    st.markdown(f'<div class="bubble {cls}">{text_to_render}</div>', unsafe_allow_html=True)

# ==========================================
# 🔹 UI: 사이드바
# ==========================================
with st.sidebar:
    st.success(f"DB 준비 완료\n(약물: {num_drugs}개, 질병: {num_diseases}개, 수술/시술: {num_procedures}개)")
    st.markdown(f"<div style='font-size:12px; margin-top:-10px; margin-bottom:10px;'><b>LLM:</b> {CHAT_MODEL}</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-card"><b>검색 옵션</b></div>', unsafe_allow_html=True)
    k_slider = st.slider("문서 검색 개수 (k)", 3, 7, 4)
    only_preg = st.checkbox("임부금기 정보 우선 (미적용)") 
    
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
    with st.spinner("답변 생성 중…"):
        bubble_container = st.empty()
        streaming_answer = "" # 원본 텍스트 (이스케이프 안됨)
        
        if context is not None:
            try:
                for chunk in summarizer_chain.stream({
                    "context": context, 
                    "question": rewritten_question,
                    "chat_history": langchain_history
                }):
                    streaming_answer += chunk
                    # 스트리밍되는 텍스트를 HTML 이스케이프 처리
                    streaming_text_escaped = html.escape(streaming_answer)
                    bubble_container.markdown(f'<div class="bubble ai">{streaming_text_escaped}▌</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ 답변 생성 중 오류: {e}")
                streaming_answer = "답변 생성 중 오류가 발생했습니다."
        else:
            streaming_answer = response_text
        
        # 최종 기본 답변은 이스케이프 처리
        base_answer_escaped = html.escape(streaming_answer)

    # 7. [Chain 4] 추천 질문 생성
    recommendations_html = "" # HTML을 저장할 변수
    
    # 1. DB 검색(context)이 성공했는가?
    context_is_success = (
        context is not None and 
        "찾지 못했습니다" not in context and 
        "자료가 없습니다" not in context
    )
    
    # 2. AI의 최종 답변(streaming_answer)이 성공했는가? (환각 방지)
    failure_keywords = ["죄송합니다", "찾지 못했습니다", "일치하는", "자료가 없습니다", "오류가 발생했습니다", "검색어는 없습니다"]
    answer_is_success = not any(keyword in streaming_answer for keyword in failure_keywords)

    # 3. 둘 다 성공해야 추천 질문 실행
    is_successful_answer = context_is_success and answer_is_success
    
    
    if is_successful_answer:
        with st.spinner("🔍 관련 질문 추천 중..."):
            try:
                # invoke에는 원본 텍스트(streaming_answer)를 사용
                recommendations_output = recommender_chain.invoke({"answer": streaming_answer})
                if "정보 없음" not in recommendations_output:
                    # 추천 질문을 HTML 형식으로 생성
                    recommendations_html = f"""
                    <div style="margin-top: 15px; font-size: 14px; border-top: 1px solid #eee; padding-top: 12px;">
                        <b style="color:var(--ink);">💡 관련 추천 질문:</b><br>
                        {recommendations_output.replace('•', '• ').replace('\n', '<br>')}
                    </div>
                    """
            except Exception as e:
                print(f"--- ⚠️ 추천 질문 생성 실패: {e} ---")

    # 8. 최종 답변 결합 및 렌더링
    final_html_answer = base_answer_escaped + recommendations_html
    
    # 9. 스트리밍이 끝난 bubble_container를 최종 HTML 답변으로 업데이트
    bubble_container.markdown(f'<div class="bubble ai">{final_html_answer}</div>', unsafe_allow_html=True)
    
    # 10. 이 최종 HTML 덩어리를 히스토리에 저장
    st.session_state.history.append(("ai", final_html_answer))


st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<footer class="note">※ 본 서비스는 정보 제공용이며, 의료 전문의 상담을 대체하지 않습니다.</footer>', unsafe_allow_html=True)