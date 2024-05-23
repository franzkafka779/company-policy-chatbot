import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from googletrans import Translator

# PDF에서 텍스트와 표 추출
def extract_text_and_tables(pdf_path):
    text = ""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            tables.extend(page.extract_tables())
    return text, tables

# 텍스트 임베딩 생성 및 FAISS 인덱스 생성
def create_faiss_index(texts):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model

def search_faiss_index(query, index, model, texts, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [texts[idx] for idx in indices[0]]

# 번역 함수 (Google Translate API 사용)
translator = Translator()

def translate(text, src='ko', dest='en'):
    return translator.translate(text, src=src, dest=dest).text

# Streamlit UI 구현
st.title("회사 내규 챗봇")

# PDF 파일에서 텍스트와 테이블 추출
pdf_path = 'data/포커스미디어_상품정책_5.4.2.pdf'
text, tables = extract_text_and_tables(pdf_path)

# 텍스트 임베딩 생성 및 FAISS 인덱스 생성
texts = text.split('\n')  # 간단히 줄 단위로 나눔
index, model = create_faiss_index(texts)

user_input = st.text_input("질문을 입력하세요:")

if user_input:
    # 질문 번역 (한국어 -> 영어)
    translated_input = translate(user_input, src='ko', dest='en')
    
    # 검색
    results = search_faiss_index(translated_input, index, model, texts)
    
    # 결과 번역 (영어 -> 한국어)
    translated_results = [translate(result, src='en', dest='ko') for result in results]
    
    # 결과 출력
    for result in translated_results:
        st.write(result)
