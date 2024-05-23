import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from googletrans import Translator
import gdown
import os

# 모델 파일이 존재하지 않는 경우 Google Drive에서 다운로드
if not os.path.exists('data/model.pkl'):
    url = 'https://drive.google.com/file/d/1_etZpxgAXnaB6gHQUCY5mfryUN3wV7As/view?usp=drive_link'  # Google Drive 파일 ID로 대체
    gdown.download(url, 'data/model.pkl', quiet=False)

# Load precomputed FAISS index, model, and texts
with open('data/index.pkl', 'rb') as f:
    index = pickle.load(f)
with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/texts.pkl', 'rb') as f:
    texts = pickle.load(f)

# 번역 함수 (Google Translate API 사용)
translator = Translator()

def translate(text, src='ko', dest='en'):
    return translator.translate(text, src=src, dest=dest).text

def search_faiss_index(query, index, model, texts, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [texts[idx] for idx in indices[0]]

# Streamlit UI 구현
st.title("회사 내규 챗봇")

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
