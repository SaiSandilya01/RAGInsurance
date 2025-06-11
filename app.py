import streamlit as st
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os


os.environ["GOOGLE_API_KEY"] = "api key"

if "qa_chain" not in st.session_state:
    # Load data
    with open("./data/insurance_policies.json", "r") as f:
        data = json.load(f)
    docs = [Document(page_content=d["content"], metadata={"source": d["id"]}) for d in data]


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_store")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


    llm = GoogleGenerativeAI(model="models/gemini-2.0-flash", convert_system_message_to_human=True)


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    st.session_state.qa_chain = qa_chain


st.title("Insurance Policy Assistant")
user_input = st.text_input("Ask a question about insurance policies:")

if user_input:

        response = st.session_state.qa_chain.invoke(user_input)
        st.success(response)
