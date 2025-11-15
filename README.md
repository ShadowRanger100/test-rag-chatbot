---

# Local PDF Chatbot

## Overview

This app lets you upload a PDF and ask questions about its content. It uses Streamlit for the interface, LM Studio for embeddings and responses, and FAISS for finding the most relevant parts of the document.

## How It Works

1. Upload a PDF from the sidebar.
2. The text is extracted and split into chunks.
3. Each chunk is converted to an embedding using your LM Studio embedding model.
4. A FAISS index stores these embeddings for fast search.
5. When you ask a question, the app finds the closest matching chunks and sends them to your LM Studio chat model to generate an answer.

## Requirements

Install the needed libraries:

```
pip install streamlit PyPDF2 numpy requests faiss-cpu langchain-text-splitters
```

## LM Studio Setup

* Load an embedding model (for example: nomic-embed-text).
* Load a chat model (for example: phi-3-mini or mistral).
* Turn on the local server.
* Update the URLs in the script if your LM Studio runs on a different IP.

## Run the App

```
streamlit run chatbot.py
```

Upload a PDF and start asking questions.
