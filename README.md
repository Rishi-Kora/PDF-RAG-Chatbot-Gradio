# PDF-RAG-Chatbot-Gradio
This Python application lets you **chat with the contents of PDF files** using a **Retrieval-Augmented Generation (RAG)** pipeline powered by **LangChain**, **OpenAI embeddings**, and a **Gradio web interface**.

## 🚀 Features
- **Automatic PDF ingestion**: Loads all `.pdf` files from the `PDFs` folder.
- **Chunking & Embeddings**: Splits documents into manageable chunks and generates embeddings with OpenAI.
- **Vector Search with Chroma**: Stores embeddings in a persistent ChromaDB for fast retrieval.
- **Natural Language Q&A**: Uses `ChatOpenAI` and `RetrievalQA` to answer questions from your PDFs.
- **Persistent Chat History**: Saves past conversations in `chat_history.json` so they’re available across sessions.
- **Interactive Web UI**: Gradio-powered chatbot interface with a “Clear Chat” button and real-time Q&A.
- **Shareable**: Option to share your Gradio app publicly.

## 🛠 How It Works
1. PDFs are loaded using **PyMuPDFLoader** from LangChain.
2. The text is split into overlapping chunks with `CharacterTextSplitter`.
3. **OpenAI embeddings** are generated and stored in **Chroma** (local persistent DB).
4. A **retrieval-based QA chain** is built using `ChatOpenAI`.
5. User queries are matched to relevant PDF sections and answered.
6. Gradio serves an easy-to-use chat interface.

## 📂 Project Structure
- PDFs # Folder containing your PDF files
- key.env # Environment variables (e.g., OpenAI API key)
- chat_history.json # Saved chat history
- pdf_rag_chatbot_gradio.py # Main app script

## ▶️ Usage
```bash
pip install -r requirements.txt
python pdf_rag_chatbot_gradio.py
```




