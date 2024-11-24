
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
import time
import streamlit as st
from streamlit_chat import message

api_key_1 = st.secrets["GROQ_API_KEY_1"]
api_key_2 = st.secrets["GROQ_API_KEY_2"]

# Streamed response emulator
def response_generator(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm_1 = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key = api_key_1 )

llm_2 = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key = api_key_2)

system_prompt = (
    "You are an assistant for medical question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. Give answers of lenghts between 300-400 words and bold the keywords."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)



def response_generate(query):
    try:
        question_answer_chain = create_stuff_documents_chain(llm_1, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result = rag_chain.invoke({"input": query})
        result_answer = result['answer']
    except:
        question_answer_chain = create_stuff_documents_chain(llm_2, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result = rag_chain.invoke({"input": query})
        result_answer = result['answer']

    formatted_response = f"{result_answer}\n\nRelevant Sources:\n"

    for i in range(3):
        src = result['context'][i].metadata['source'].split('/')[-1].strip(".pdf")
        page = result['context'][i].metadata['page']
        page = str(int(page)+1)
        formatted_response += f"- Source: {src}, Page: {page}\n"

    return formatted_response, result_answer

def  main():
    st.set_page_config(layout="wide")

    # if "history" not in st.session_state:
    #     st.session_state.history = "" 
    
    
    st.markdown("<h1 style='text-align: center; color: blue;'> TextBook Chatbot </h1>", unsafe_allow_html=True)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Update sales history in session_state
        # st.session_state.history += "\nUser: " + prompt

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            formatted_response, result_answer = response_generate(prompt)
            output = formatted_response.lstrip("\n")
            response = st.write_stream(response_generator(output))

            

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Update sales history with assistant response
        # st.session_state.history += "\nAI: " + result_answer

from streamlit.web import cli as stcli
from streamlit import runtime
import sys

if __name__ == '__main__':
    main()      
