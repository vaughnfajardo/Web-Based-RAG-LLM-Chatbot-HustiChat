import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document



load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")


def get_vectorstore_from_url(gr_number, year, month):

    month_abbreviations = {
    "January": "jan",
    "February": "feb",
    "March": "mar",
    "April": "apr",
    "May": "may",
    "June": "jun",
    "July": "jul",
    "August": "aug",
    "September": "sep",
    "October": "oct",
    "November": "nov",
    "December": "dec" }

    month_abbr = month_abbreviations[month]

    url = f"https://lawphil.net/judjuris/juri{year}/{month_abbr}{year}/gr_{gr_number}_{year}.html"

    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Based on the previous discussion, formulate a search query to find related information.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain


def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Respond to the user's queries using the following context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']





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