import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import gradio as gr

load_dotenv("key.env")

PDF_DIR = "PDFs"
HISTORY_FILE = "chat_history.json"
initial_greeting = [["Hello! How can I help you?", ""]]


# Load PDFs
documents = []
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        loader = PyMuPDFLoader(os.path.join(PDF_DIR, filename))
        documents.extend(loader.load())

if not documents:
    raise ValueError("❌ No valid PDFs found in the folder.")


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()


qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)


def save_history_to_file(history, path=HISTORY_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def chat_with_pdf(query, history):
    response = qa_chain.run(query)
    history = history + [(query, response)]
    save_history_to_file(history)
    return history, history


def clear_chat():
    return initial_greeting, initial_greeting, ""

# Load history
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        initial_history = json.load(f)
        # Ensure all items are proper list of two strings
        initial_history = [
            list(pair) for pair in initial_history
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(p, str) for p in pair)
        ]
else:
    initial_history = []


with gr.Blocks() as demo:
    gr.Markdown("## Chat with Your PDFs")
    chatbot = gr.Chatbot(value=initial_history, label="Assistant")
    message = gr.Textbox(placeholder="Ask something...", label="Your Question")
    state = gr.State(initial_history)
    clear_btn = gr.Button("Clear Chat")

    message.submit(chat_with_pdf, [message, state], [chatbot, state])
    message.submit(lambda: "", None, message)  

    clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot, state, message])

demo.launch(share=True)
