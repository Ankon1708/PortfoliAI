import RAG_Mod

import streamlit as st

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

if query:
    result=RAG_Mod.My_Chatbot(query)
    st.header("Answer")
    st.write(result)
