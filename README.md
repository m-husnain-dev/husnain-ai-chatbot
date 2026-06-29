# Husnain's Personal RAG Chatbot
# 🤖 Husnain AI — Personal RAG Chatbot

A personal AI chatbot built from scratch using **Retrieval Augmented Generation (RAG)** architecture. Ask it anything about me — skills, projects, goals — and it answers from my own knowledge base.

---

## 🚀 Live Demo

> Run locally — see setup below

---

## 🧠 How It Works

```
Your Question → Embeddings → Vector Search (FAISS) → Relevant Context → LLaMA 3.1 → Answer
```

1. Documents from `data/` folder are chunked and embedded
2. Stored in a local FAISS vector database
3. User query is matched against stored embeddings
4. Top relevant chunks sent to Groq LLaMA 3.1
5. LLM generates a context-aware answer

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| LangChain | RAG pipeline |
| FAISS | Vector database |
| FastEmbed | Embeddings (local, free) |
| Groq + LLaMA 3.1 | LLM (free API) |
| Streamlit | Chat UI |

---

## ✨ Features

- Neon dark UI with avatar
- Fixed header with profile
- Supports `.txt` and `.pdf` knowledge base
- Add any document to `data/` — bot learns automatically
- Free to run (no paid APIs required beyond Groq free tier)

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/m-husnain-dev/husnain-ai-chatbot.git
cd husnain-ai-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get free Groq API key
Sign up at [console.groq.com](https://console.groq.com) → Create API Key

### 4. Create `.env` file
```
GROQ_API_KEY=your_key_here
```

### 5. Run
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
husnain-ai-chatbot/
├── app.py              # Main chatbot application
├── requirements.txt    # Python dependencies
├── data/
│   ├── about_me.txt    # Personal knowledge base
│   └── avatar.jpg      # Profile picture
└── .gitignore
```

---

## 👨‍💻 About Me

**Muhammad Husnain** — BS Artificial Intelligence @ NTU Faisalabad  
Building AI tools that make technology accessible for Pakistani & Indian students.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/muhammad-husnain-fareed)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/m-husnain-dev)
