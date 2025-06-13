# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 01:03:35 2025

@author: ASUS

import os
os.chdir('C:\\Users\\ASUS\\Documents\\Python Scripts\\LnE Langchain')
"""
from Keys import my_google_api_key
os.environ["GOOGLE_API_KEY"] = my_google_api_key

from langchain_google_genai import ChatGoogleGenerativeAI
import langchain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_db=FAISS.load_local("faiss_vector_db",embeddings,allow_dangerous_deserialization=True)

import streamlit as st

st.title("PortfoliAI")
st.sidebar.title("I am Ankon Bhowmick, an aspiring data scientist.\n This is an RAG-based chat bot that demonstrates my new-found skills in Generative AI.\n This bot is capable of answering questions about my academic journey, internship experience, and my entire portfolio of projects and skills.\n Feel free to ask any question pertaining to them.")

main_placeholder = st.empty()
query = main_placeholder.text_input("Question: ")
sys_prompt='''You are an AI assistant that answers queries about Ankon Bhowmick's academic and internship experience. 
            Answer the questions while framing the candidate's experience and skills in the most positive way 
            but do not make up any information beyond what is provided.'''
if query:
    query=sys_prompt+query
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_db.as_retriever())
    result=chain({"question": query}, return_only_outputs=True)
    st.header("Answer")
    st.write(result["answer"])
    

