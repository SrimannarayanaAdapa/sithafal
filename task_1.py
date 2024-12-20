import os,pickle
import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Streamlit Application for Interactive PDF Analysis
st.title("PDF Analyzer and Query Tool📄")
st.sidebar.header("Upload📤 and Analyze🔍 PDFs")

# Upload Section
pdf_files = st.sidebar.file_uploader(
    "Choose PDF files for analysis", type=["pdf"], accept_multiple_files=True
)
start_analysis = st.sidebar.button("Analyze PDFs 📈")
vector_store_path = "pdf_query_index.pkl"  # File for saving vector store

# Initialize LLM with API Key
groq_api_key = "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"  
llm_model = ChatGroq(temperature=0.65, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Analyze PDFs
if start_analysis:
    if pdf_files:
        st.sidebar.info("Processing uploaded PDFs ⚙️🔍")
        combined_text = ""

        for file in pdf_files:
            text_content = extract_text(file)
            combined_text += text_content + "\n"

        # Splitting text for better vectorization
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=30)
        document_chunks = text_splitter.split_text(combined_text)

        # Create embeddings and vector store
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_index = FAISS.from_texts(document_chunks, embedding_model)

        # Save vector store
        with open(vector_store_path, "wb") as file:
            pickle.dump(vector_index, file)

        st.sidebar.success("PDFs analyzed and vector index created!")
    else:
        st.sidebar.error("Please upload at least one PDF file.")

# Query Section
query_input = st.text_input("Ask something about your PDFs:")
if query_input:
    if os.path.exists(vector_store_path):
        # Load vector store for querying
        with open(vector_store_path, "rb") as file:
            vector_index = pickle.load(file)

        retriever = vector_index.as_retriever()
        question_answer_chain = RetrievalQA.from_chain_type(llm=llm_model, retriever=retriever)

        # Fetch and display the response
        response = question_answer_chain.run(query_input)
        st.write("### Response:")
        st.write(response)
    else:
        st.error("No vector store found. Please analyze PDFs first.")


