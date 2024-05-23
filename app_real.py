import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import os

# 텍스트 추출 함수
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# 텍스트 분할 함수
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 벡터 DB 생성 함수
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# LLM 모델 로드 함수
def load_llm_model():
    model = pipeline("text-generation", model="skt/kogpt2-base-v2", tokenizer="skt/kogpt2-base-v2")
    return model

# LLM을 사용한 답변 생성 함수
def generate_answer(llm_model, query, context):
    input_text = f"질문: {query}\n\n맥락: {context}\n\n답변:"
    answer = llm_model(input_text, max_new_tokens=100, do_sample=True, top_p=0.95, top_k=50)
    return answer[0]['generated_text']

# Streamlit UI 구현
st.title("회사 내규 챗봇")

pdf_path = 'data/포커스미디어_상품정책_5.4.2.pdf'

if os.path.exists(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(pdf_text)
    vectorstore = create_vectorstore(text_chunks)
    llm_model = load_llm_model()

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # 검색
        search_results = vectorstore.similarity_search(user_input, k=5)
        
        # 검색 결과를 하나의 컨텍스트로 합치기
        context = "\n".join([result.page_content for result in search_results])
        
        # LLM을 사용하여 답변 생성
        answer = generate_answer(llm_model, user_input, context)
        
        # 결과 출력
        st.write(answer)
else:
    st.error("PDF 파일을 data 폴더에 업로드해주세요.")
