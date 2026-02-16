import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter


# --- Page Config ---
st.set_page_config(page_title="AI Assistant with Memory", layout="wide")
st.title("🧠 AI Assistant with Chat Memory")

# Initialize Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Zone")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if uploaded_file:
    temp_file = "temp_document_memory.pdf"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Build AI Knowledge"):
        with st.spinner("Analyzing PDF..."):
            try:
                st.session_state.vectorstore = process_pdf(temp_file)
                st.success("Ready with Memory!")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Chat Interface ---
# Display previous messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Handle new user input
user_input = st.chat_input("Ask about your document...")

if user_input and "vectorstore" in st.session_state:
    # Show user message immediately
    with st.chat_message("Human"):
        st.markdown(user_input)
    
    # Setup LLM & Chain with Memory
    llm = ChatOllama(model="llama3.2:3b")
    retriever = st.session_state.vectorstore.as_retriever()
    
    # Prompt with History
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context below and the conversation history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", "Context: {context}"),
        ("human", "{question}"),
    ])

    # 1. دالة لتنسيق النصوص المستخرجة
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 2. بناء السلسلة (Chain) بوضوح تام
    # نستخدم itemgetter لضمان سحب النص فقط قبل إرساله للـ retriever

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 3. الحصول على الإجابة
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            try:
                # نمرر القاموس هنا، والـ itemgetter بالأعلى سيتكفل بالباقي
                response = rag_chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history
                })
                st.markdown(response)
                
                # تحديث الذاكرة بعد النجاح
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
elif user_input and "vectorstore" not in st.session_state:
    st.warning("Please upload and analyze a PDF first!")