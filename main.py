from Keys import my_google_api_key
import os
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
import time

st.title("PortfoliAI")
github_url = "https://github.com/Ankon1708"
linkedin_url = "https://www.linkedin.com/in/ankon-bhowmick-2a8805193/"

st.sidebar.title("I am Ankon Bhowmick, an aspiring data scientist.\n This is an RAG-based chat bot that demonstrates my new-found skills in Generative AI.\n This bot is capable of answering questions about my academic journey, internship experience, and my entire portfolio of projects and skills.\n Feel free to ask any question pertaining to them.")
st.sidebar.markdown("### 🔗 Connect")
st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?logo=github)]({github_url})")
st.sidebar.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)]({linkedin_url})")
with open("Ankon Bhowmick Resume.pdf", "rb") as file:
    st.sidebar.download_button("Download CV", file, file_name="CV.pdf")

main_placeholder = st.empty()
query = main_placeholder.text_input("Question: ")
sys_prompt='''You are an AI assistant that answers queries about Ankon Bhowmick's academic and internship experience. 
            Answer the questions while framing the candidate's experience and skills in the most positive way 
            but do not make up any information beyond what is provided.'''
if query:
    progress_bar = st.progress(0)
    status = st.empty()
    
    query=sys_prompt+query
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_db.as_retriever())

    status.text("🔍 Step 1: Retrieving relevant documents...")
    for i in range(10):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
        
    result=chain({"question": query}, return_only_outputs=True)

    status.text("🧩 Step 2: Mapping documents and extracting information...")
    for i in range(10, 40):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    status.text("📝 Step 3: Generating final answer using LLM...")
    for i in range(40, 100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)
        
    st.header("Answer")
    st.write(result["answer"])
    

