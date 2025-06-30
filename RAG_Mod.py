# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 20:26:24 2025

@author: ASUS
"""

#%% Import API Keys, libraries, set environment for LangSmith

from Keys import my_lang_api_key, my_google_api_key
import os
os.environ["GOOGLE_API_KEY"] = my_google_api_key
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_PROJECT']="PortfoliAI_v1"
os.environ['LANGCHAIN_API_KEY'] = my_lang_api_key

import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.runnables.base import RunnableEach
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from IPython.display import Markdown


#%% Initialise LLM and embeddings, import VDB

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_db=FAISS.load_local("faiss_vector_db",embeddings,allow_dangerous_deserialization=True)


#%% Define Prompts

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that answers queries about a candidate."
)

prompt_rag_human = HumanMessagePromptTemplate.from_template(
    """You are tasked with drafting an answer for a question asked regarding a candidate's skills and experience.
The answer should only contain information on the candidate in the form of context that has been made available to you, do not falsify any skills or experience. 
For behavourial questions, use slightly creative vocabulary to best answer them.
The context is: {context}
The question is: {question}""",
    input_variables=["question"]
)

prompt_rag = ChatPromptTemplate.from_messages([system_prompt, prompt_rag_human])

prompt_genq_human = HumanMessagePromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
)

prompt_genq = ChatPromptTemplate.from_messages([system_prompt, prompt_genq_human])

prompt_final_human=HumanMessagePromptTemplate.from_template('''
You are given information about the candidate that is relevant to the question. Here's the information : {context}
Answer this question about the candidate by coherently using the information while trying to reduce redundancy. Remember that this question is asked by someone else about the candidate : {question}''')

prompt_final=ChatPromptTemplate.from_messages([system_prompt, prompt_final_human])


#%% Define Retriever and Chains

retriever=vector_db.as_retriever(search_kwargs={"k": 3})

# Basic RAG Chain using a query, and a retriever for context
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

# Generate multiple queries for wider semantic search
generate_queries = (
    prompt_genq
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# Run the generated queries individually through rag chain and combine all answers
parallel_rag=(
    RunnableEach(bound=rag_chain)
    | (lambda x: "".join(x))
)

# CC: Complete Chain
# Use combined answer as context and generate a final answer
CC_RAG=(                                
    {'context':(generate_queries
               | parallel_rag),
     'question':RunnablePassthrough()
    }
    | prompt_final
    | llm
    | StrOutputParser()
)


#%% Define function for modular use

def My_Chatbot(query):
    return CC_RAG.invoke(query)
