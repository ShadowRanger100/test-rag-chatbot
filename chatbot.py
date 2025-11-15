import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import numpy as np
import faiss

# --- LM Studio local API endpoints ---
EMBEDDING_URL = "http://192.168.56.1:1234/v1/embeddings"   # Embedding model (e.g., nomic-embed-text-v1.5)
CHAT_URL = "http://192.168.56.1:1234/v1/chat/completions"  # Chat model (e.g., phi-3 or mistral)

# --- Streamlit UI ---
st.header("💬 My Local PDF Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")


# --- Function to get embedding from LM Studio ---
def get_embedding(text):
    """Send text to LM Studio embedding model and get the embedding vector"""
    response = requests.post(
        EMBEDDING_URL,
        headers={"Content-Type": "application/json"},
        json={"input": text}
    )
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        st.error(f"Embedding Error: {response.text}")
        return None


# --- If a file is uploaded ---
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # --- Split text into chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.write(f"✅ Split PDF into {len(chunks)} chunks")

    # --- Generate embeddings for all chunks ---
    st.write("Generating embeddings... ⏳")
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb is not None:
            embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")
    st.success("✅ Embeddings generated successfully!")

    # --- Create FAISS index ---
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    st.success("✅ FAISS index created!")

    # --- User query input ---
    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        # 1️⃣ Get embedding for the query
        query_emb = np.array([get_embedding(user_question)]).astype("float32")

        # 2️⃣ Search FAISS for most similar chunks
        D, I = index.search(query_emb, k=3)
        context = "\n\n".join([chunks[i] for i in I[0]])

        # 3️⃣ Send context + question to LM Studio chat model
        payload = {
            "model": "phi-3-mini-4k-instruct",  # change this to your loaded chat model name
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers questions using only the given context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
            ],
            "temperature": 0.2
        }

        response = requests.post(CHAT_URL, headers={"Content-Type": "application/json"}, json=payload)

        # 4️⃣ Display answer
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            st.subheader("💡 Answer:")
            st.write(answer)
        else:
            st.error(f"Chat Model Error: {response.text}")
