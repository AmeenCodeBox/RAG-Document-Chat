# 🤖 Local RAG AI Assistant for PDFs with (Llama 3.2 3B)

An intelligent, privacy-focused PDF Chatbot that runs entirely on your local machine. No data is sent to the cloud.

## 🌟 Key Features
- **Privacy-First**: All processing happens locally via Ollama.
- **RAG Architecture**: Uses Retrieval-Augmented Generation for accurate PDF analysis.
- **Multilingual**: Capable of understanding and responding in English, German, and Arabic and more.

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **AI Orchestration**: LangChain
- **LLM**: Ollama (Llama 3.2 3B)
- **Vector Store**: ChromaDB

## 🚀 Getting Started
1. **Install Ollama**: Download from [ollama.com](https://ollama.com)
2. **Download Model**: `ollama pull llama3.2:3b`
3. **Setup Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
4. **Run App**: 
    ```bash
    streamlit run app.py
