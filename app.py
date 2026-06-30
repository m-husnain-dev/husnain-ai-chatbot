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

st.set_page_config(page_title="Husnain AI", page_icon="🤖", layout="centered")

# ─── AVATAR ───────────────────────────────────────────────────────────────────

def get_avatar_b64():
    path = os.path.join(DATA_FOLDER, "avatar.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

avatar_b64 = get_avatar_b64()
avatar_html = (
    f'<img class="av-img" src="data:image/jpeg;base64,{avatar_b64}"/>'
    if avatar_b64 else
    '<div class="av-img av-fallback">🧑‍💻</div>'
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

.stApp { background: #ffffff !important; font-family: 'Inter', sans-serif !important; }
.stApp > div { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container { max-width: 600px !important; padding: 0 20px 40px !important; }
#MainMenu, footer { display: none !important; }

/* Blobs */
.blobs { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 0; overflow: hidden; }
.blob { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.5; animation: bfloat 10s ease-in-out infinite; }
.b1 { width: 500px; height: 500px; background: radial-gradient(#ffb5e8, #ff9de2); top: -150px; left: -150px; animation-delay: 0s; }
.b2 { width: 400px; height: 400px; background: radial-gradient(#b5deff, #85c1ff); top: 40%; right: -100px; animation-delay: -3s; }
.b3 { width: 350px; height: 350px; background: radial-gradient(#85f5c0, #5ce8a0); bottom: -100px; left: 35%; animation-delay: -6s; }
.b4 { width: 280px; height: 280px; background: radial-gradient(#ffdfba, #ffcc88); top: 25%; left: 5%; animation-delay: -2s; }
.b5 { width: 220px; height: 220px; background: radial-gradient(#e8b5ff, #d485ff); bottom: 15%; right: 10%; animation-delay: -4s; }
@keyframes bfloat { 0%,100% { transform: translate(0,0) scale(1); } 33% { transform: translate(40px,-40px) scale(1.08); } 66% { transform: translate(-30px,30px) scale(0.93); } }

/* Portfolio header */
.port-logo { position: fixed; top: 20px; left: 50%; transform: translateX(-50%); width: 44px; height: 44px; background: #1a1a1a; border-radius: 12px; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 900; font-size: 20px; z-index: 1000; font-family: 'Inter', sans-serif; }
.port-greet { font-size: 18px; color: #555; font-weight: 500; text-align: center; margin-bottom: 4px; position: relative; z-index: 100; }
.port-title { font-size: clamp(48px, 9vw, 80px); font-weight: 900; color: #1a1a1a !important; letter-spacing: -4px; line-height: 0.95; text-align: center; margin-bottom: 28px; position: relative; z-index: 100; }

/* Avatar */
.av-outer { position: relative; width: 200px; height: 200px; margin: 0 auto 28px; z-index: 100; }
.av-ring { position: absolute; inset: -6px; border-radius: 50%; background: conic-gradient(from 0deg, #ffb5e8, #b5deff, #85f5c0, #ffdfba, #e8b5ff, #ffb5e8); animation: avspin 5s linear infinite; filter: blur(2px); opacity: 0.75; }
@keyframes avspin { to { transform: rotate(360deg); } }
.av-img { position: relative; z-index: 1; width: 200px; height: 200px; border-radius: 50%; object-fit: cover; object-position: center top; border: 4px solid #fff; box-shadow: 0 24px 64px rgba(0,0,0,0.12); display: block; }
.av-fallback { position: relative; z-index: 1; width: 200px; height: 200px; border-radius: 50%; background: linear-gradient(135deg, #e8f4ff, #f5e8ff); display: flex; align-items: center; justify-content: center; font-size: 90px; border: 4px solid #fff; box-shadow: 0 24px 64px rgba(0,0,0,0.12); }

/* Streamlit quick buttons */
div[data-testid="stHorizontalBlock"] { gap: 10px !important; justify-content: center; position: relative; z-index: 100; }
div[data-testid="stHorizontalBlock"] button {
    background: rgba(255,255,255,0.85) !important;
    border: 1.5px solid rgba(0,0,0,0.08) !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important;
    color: #333 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 14px 10px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
div[data-testid="stHorizontalBlock"] button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 30px rgba(0,0,0,0.1) !important;
    background: rgba(255,255,255,0.95) !important;
}

/* Chat messages */
div[data-testid="stChatMessage"] { background: rgba(255,255,255,0.95) !important; border: 1.5px solid rgba(0,0,0,0.07) !important; border-radius: 18px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important; margin-bottom: 10px !important; }
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] div,
div[data-testid="stChatMessage"] ol,
div[data-testid="stChatMessage"] ul,
div[data-testid="stChatMessage"] strong { color: #222 !important; font-family: 'Inter', sans-serif !important; opacity: 1 !important; }
div[data-testid="stChatMessageContent"] { background: transparent !important; }
.stChatInput textarea { background: rgba(255,255,255,0.95) !important; border: 1.5px solid rgba(0,0,0,0.12) !important; border-radius: 56px !important; font-family: 'Inter', sans-serif !important; color: #1a1a1a !important; }
.stChatInput textarea::placeholder { color: #888 !important; opacity: 1 !important; }
.stChatInput { background: transparent !important; }
div[data-testid="stChatInput"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─── HTML HEADER ──────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="blobs">
  <div class="blob b1"></div><div class="blob b2"></div>
  <div class="blob b3"></div><div class="blob b4"></div><div class="blob b5"></div>
</div>
<div class="port-logo">H</div>
<br/><br/><br/>
<p class="port-greet">Hey, I'm Husnain 👋</p>
<h1 class="port-title">AI Portfolio</h1>
<div class="av-outer">
  <div class="av-ring"></div>
  {avatar_html}
</div>
""", unsafe_allow_html=True)

# ─── QUICK BUTTONS (Streamlit native) ─────────────────────────────────────────

if "quick_input" not in st.session_state:
    st.session_state.quick_input = ""

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("😊\nMe"):
        st.session_state.quick_input = "Tell me about Husnain"
        st.rerun()
with col2:
    if st.button("💼\nProjects"):
        st.session_state.quick_input = "What are your projects?"
        st.rerun()
with col3:
    if st.button("⚡\nSkills"):
        st.session_state.quick_input = "What are your skills?"
        st.rerun()
with col4:
    if st.button("📬\nContact"):
        st.session_state.quick_input = "How can I contact Husnain?"
        st.rerun()

st.markdown("<br/>", unsafe_allow_html=True)

# ─── VECTORSTORE ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
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
    embeddings  = FastEmbedEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

@st.cache_resource(show_spinner=False)
def get_chain(_vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        "You are Husnain's personal AI assistant, speaking to a visitor about Husnain.\n"
        "Always refer to Husnain in the third person (he, his, Husnain) — never say 'you' or 'your', "
        "since the visitor is asking ABOUT Husnain, not about themselves.\n"
        "Answer using the context below. If the info is not available, say: This information is not available.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

# ─── CHAT ─────────────────────────────────────────────────────────────────────

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing.")
    st.stop()

vectorstore = build_vectorstore()
chain       = get_chain(vectorstore)

AVATARS = {"user": "🧑", "assistant": "🤖"}

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.write(msg["content"])

# Handle quick button input
if st.session_state.quick_input:
    q = st.session_state.quick_input
    st.session_state.quick_input = ""
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.write(q)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Thinking..."):
            response = chain.invoke(q)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.write(user_input)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Thinking..."):
            response = chain.invoke(user_input)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
