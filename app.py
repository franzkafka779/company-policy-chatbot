import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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

# Streamlit UI 구현
st.title("회사 내규 챗봇")

pdf_path = 'data/포커스미디어_상품정책_5.4.2.pdf'

if os.path.exists(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(pdf_text)
    vectorstore = create_vectorstore(text_chunks)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # 검색
        search_results = vectorstore.similarity_search(user_input, k=5)
        
        # 결과 출력
        for result in search_results:
            st.write(result.page_content)
else:
    st.error("PDF 파일을 data 폴더에 업로드해주세요.")
