import os
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DATA_FOLDER  = "data"

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Husnain AI", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse at 20% 50%, #0d0030 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, #001a1a 0%, transparent 50%);
}
h1 {
    color: #00ffff !important;
    text-shadow: 0 0 10px #00ffff, 0 0 30px #00ffff, 0 0 60px #00aaff;
    text-align: center;
}
.stChatInput textarea {
    background: #111 !important;
    border: 1px solid #00ffff !important;
    color: #fff !important;
    box-shadow: 0 0 8px #00ffff55;
}
div[data-testid="stChatMessageContent"] p { color: #e0e0e0 !important; }
hr { border-color: #00ffff33 !important; }
.avatar-wrapper {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: #0a0a0f;
    border-bottom: 1px solid #00ffff22;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 10px 0;
    backdrop-filter: blur(10px);
}
.avatar-img {
    width: 40px !important;
    height: 40px !important;
}
.stApp > div {
    padding-top: 80px !important;
}
.avatar-img {
    width: 90px; height: 90px; border-radius: 50%;
    border: 2px solid #00ffff;
    box-shadow: 0 0 16px #00ffff, 0 0 40px #00aaff55;
    object-fit: cover;
}
.avatar-name {
    color: #00ffff; font-size: 14px; margin-top: 8px;
    text-shadow: 0 0 8px #00ffff; letter-spacing: 1px;
}
.neon-badge { color: #888; font-size: 11px; margin-top: 2px; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #00ffff44; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── VECTORSTORE ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Knowledge base loading...")
def build_vectorstore():
    docs = []
    for filename in os.listdir(DATA_FOLDER):
        filepath = os.path.join(DATA_FOLDER, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        else:
            continue
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)

    embeddings = FastEmbedEmbeddings()
    vectorstore  = FAISS.from_documents(chunks, embeddings)
    return vectorstore


@st.cache_resource(show_spinner=False)
def get_chain(_vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
You are Husnain's personal assistant.
Answer based on the context provided below. If the information is not available, say: "I do not have this information.""

Context: {context}

Query: {question}
Answer:""")

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─── AVATAR ───────────────────────────────────────────────────────────────────

def get_avatar_html():
    avatar_path = os.path.join(DATA_FOLDER, "avatar.jpg")
    if os.path.exists(avatar_path):
        with open(avatar_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'<img class="avatar-img" src="data:image/jpeg;base64,{data}"/>'
    return '''<div class="avatar-img" style="background:linear-gradient(135deg,#001a2e,#003366);
        display:flex;align-items:center;justify-content:center;
        font-size:28px;font-weight:700;color:#00ffff;
        text-shadow:0 0 10px #00ffff;">H</div>'''


# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("🤖 Husnain AI")

st.markdown(f"""
<div class="avatar-wrapper">
    {get_avatar_html()}
    <div class="avatar-name">HUSNAIN</div>
    <div class="neon-badge"> AI/ML Engineer  ·  Faisalabad</div>
</div>
""", unsafe_allow_html=True)

st.divider()

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY ")
    st.stop()

vectorstore = build_vectorstore()
chain       = get_chain(vectorstore)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Anything you want to know about me...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            response = chain.invoke(user_input)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
