import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv(dotenv_path="D:\PYTHON\GENERATIVE AI\.ENV FILES\.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

Model = ChatGroq(api_key=GROQ_API_KEY, model= "llama-3.1-8b-instant")

Prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent, helpful, and friendly chatbot designed to assist users with a wide range of questions and tasks. Your role is to provide accurate, concise, and relevant responses while maintaining a professional yet approachable tone. Follow these guidelines:
1. Understand the user's intent and context before responding.
2. Provide clear, accurate, and helpful answers, prioritizing user satisfaction.
3. Use a conversational and engaging tone, avoiding overly technical jargon unless requested.
4. If the query is ambiguous, ask clarifying questions to ensure the response meets the user's needs.
5. Stay neutral, respectful, and inclusive in all interactions.
6. If you don't know the answer, admit it and offer to search for relevant information or suggest alternatives.
7. Avoid generating or reproducing harmful, offensive, or misleading content.
8. When appropriate, incorporate humor or creativity to enhance the user experience, but keep it relevant and tasteful.

Context: {context}

Question: {input}
    """)
])

def create_vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embedding = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader("Research Papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)

user_Prompt = st.text_input("Enter your question from the Research Papers")

if st.button("Do Document Embeddings"):
    create_vector_embedding()
    st.write("Vector Database is Ready")

if user_Prompt:
    if "vector" not in st.session_state:
        st.error("Please create document embeddings first by clicking 'Do Document Embeddings'.")
    else:
        try:
            Document_chain = create_stuff_documents_chain(Model, Prompt)
            Retriever = st.session_state.vector.as_retriever()
            Retrieval_chain = create_retrieval_chain(Retriever, Document_chain)
            Response = Retrieval_chain.invoke({"input": user_Prompt})
            st.write(Response["answer"])
            with st.expander("Similarity Search"):
                for i, Doc in enumerate(Response["context"]):
                    st.write(Doc.page_content)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
else:
    with st.expander("Similarity Search"):
        st.write("No context available. Please submit a query after creating embeddings.")


