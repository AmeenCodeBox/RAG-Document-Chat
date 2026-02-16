import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="AI Document Researcher", layout="wide")
st.title("🔍 AI Document Researcher")
st.markdown("### Extract precise information from your PDF files locally.")

# --- Document Processing Logic ---
def process_pdf(file_path):
    """
    Loads a PDF, splits it into chunks, and creates a local vector store.
    """
    # Load the document
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Split text into manageable chunks for the LLM
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # Initialize local embeddings using Ollama
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    
    # Create and return the vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file:
        # Save the uploaded file temporarily
        temp_file = "temp_research_data.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Process & Index"):
            with st.spinner("Analyzing document..."):
                try:
                    # Store the vectorstore in session state to persist it
                    st.session_state.vectorstore = process_pdf(temp_file)
                    st.success("Document indexed successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- Main Query Interface ---
user_query = st.text_input("Enter your question about the document:")

if user_query:
    if "vectorstore" in st.session_state:
        # Initialize the LLM (Temperature 0 for factual consistency)
        llm = ChatOllama(model="llama3.2:3b", temperature=0)
        retriever = st.session_state.vectorstore.as_retriever()
        
        # Define the Prompt Template
        template = """
        You are a precise research assistant. Use the following context to answer the question.
        If the answer is not contained within the context, clearly state that the information is missing.
        
        Context: {context}
        Question: {question}
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Helper function to join retrieved document contents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Construct the LCEL Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Execute the chain and display the result
        with st.spinner("Searching for answers..."):
            result = rag_chain.invoke(user_query)
            st.markdown("#### Response:")
            st.info(result)
    else:
        st.warning("Please upload and process a document first.")