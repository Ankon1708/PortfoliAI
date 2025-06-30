import RAG_Mod

import streamlit as st

st.title("PortfoliAI")
st.sidebar.title("I am Ankon Bhowmick, an aspiring data scientist.\n This is an RAG-based chat bot that demonstrates my new-found skills in Generative AI.\n This bot is capable of answering questions about my academic journey, internship experience, and my entire portfolio of projects and skills.\n Feel free to ask any question pertaining to them.")

main_placeholder = st.empty()
query = main_placeholder.text_input("Question: ")

if query:
    result=RAG_Mod.My_Chatbot(query)
    st.header("Answer")
    st.write(result)