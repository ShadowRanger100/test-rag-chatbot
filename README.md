# 📄 LLM-Based Document Question Answering System

## 📌 Overview
This project is a local document question-answering system that allows users to upload PDF files and query their contents using a large language model. The application focuses on retrieval-based question answering by combining vector search with controlled LLM responses.

The system is designed to demonstrate document processing, semantic search, and LLM integration using a fully local setup.

---

## 🛠 Tools & Technologies
- Python  
- Streamlit  
- LM Studio (local LLM inference)  
- FAISS (vector similarity search)  
- PyPDF2  
- NumPy  
- LangChain Text Splitters  

---

## ⚙️ System Architecture
1. A PDF document is uploaded through the Streamlit interface  
2. Text is extracted and split into smaller chunks  
3. Each chunk is converted into vector embeddings using a local embedding model  
4. Embeddings are stored in a FAISS index for efficient similarity search  
5. User queries are matched against relevant document chunks  
6. Retrieved context is passed to a local LLM to generate responses  

---

## 🔍 Key Features
- Local, offline-friendly document analysis  
- Vector-based semantic search using FAISS  
- Retrieval-augmented responses to reduce hallucinations  
- Simple and intuitive UI built with Streamlit  

---

## 🚀 How to Run the Application

### Install Dependencies
```bash
pip install streamlit PyPDF2 numpy requests faiss-cpu langchain-text-splitters
