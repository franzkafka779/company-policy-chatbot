import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from googletrans import Translator
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

# 번역 함수 (Google Translate API 사용)
translator = Translator()

def translate(text, src='ko', dest='en'):
    return translator.translate(text, src=src, dest=dest).text

# Streamlit UI 구현
st.title("회사 내규 챗봇")

pdf_path = 'data/포커스미디어_상품정책_5.4.2.pdf'

if os.path.exists(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(pdf_text)
    vectorstore = create_vectorstore(text_chunks)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # 질문 번역 (한국어 -> 영어)
        translated_input = translate(user_input, src='ko', dest='en')
        
        # 검색
        search_results = vectorstore.similarity_search(translated_input, k=5)
        
        # 결과 번역 (영어 -> 한국어)
        translated_results = [translate(result.text, src='en', dest='ko') for result in search_results]
        
        # 결과 출력
        for result in translated_results:
            st.write(result)
else:
    st.error("PDF 파일을 data 폴더에 업로드해주세요.")
