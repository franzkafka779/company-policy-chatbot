import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def extract_text_and_tables(pdf_path):
    text = ""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            tables.extend(page.extract_tables())
    return text, tables

def create_faiss_index(texts):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, model, embeddings

pdf_path = 'data/포커스미디어_상품정책_5.4.2.pdf'
text, tables = extract_text_and_tables(pdf_path)
texts = text.split('\n')

index, model, embeddings = create_faiss_index(texts)

# Save the index, model, and texts for later use
with open('data/index.pkl', 'wb') as f:
    pickle.dump(index, f)
with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('data/texts.pkl', 'wb') as f:
    pickle.dump(texts, f)
