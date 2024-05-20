from qna_chatbot import get_vectorstore_from_url, get_context_retriever_chain, get_conversational_rag_chain, get_response
import streamlit as st
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough

st.set_page_config(page_title="HustiChat", page_icon="🧑‍⚖️")
st.title("")

years = list(range(1910, 2025))  # Example range from 1990 to 2024
months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]

with st.sidebar:
    st.header("HustiChat")
    gr_number_val = st.text_input("GR Number")
    month_val = st.selectbox("Month", options=months)
    year_val = st.selectbox("Year", options=years)
    start_conversation = st.button("Start Conversation")


if start_conversation:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
    if "vector_store" not in st.session_state:
        try:
            st.session_state.vector_store = get_vectorstore_from_url(gr_number_val, year_val, month_val)
        except Exception as e:
            st.error(f"Exceeded current quota.")   

if start_conversation:
    user_query = st.chat_input("Type your message here...")
    if user_query:
        try:
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            if 'insufficient_quota' in str(e):
                st.error(f"Exceeded current quota.")  

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("HustiChat"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

st.session_state['conversation_started'] = start_conversation