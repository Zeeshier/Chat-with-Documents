import os
import streamlit as st
from itertools import zip_longest
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from operator import itemgetter
import pinecone

# Streamlit Page Configuration
st.set_page_config(page_title="Document", page_icon="🔬", layout="wide")

# Sidebar for API Key Input
st.sidebar.header("Chat with Your Documents")
st.sidebar.write("### Enter Your API Keys and Upload Documents")
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")

# Initialize session state variables
if 'documents' not in st.session_state:
    st.session_state['documents'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []  # Conversation history

# File uploader for PDF documents
uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)

# Process and Load Documents
if uploaded_files and google_api_key and pinecone_api_key:
    pages = []
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        pages.extend(loader.load())
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(pages)

    # Set up embeddings and Pinecone
    embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    index_name = "ai"
    pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")

    # Store embeddings in Pinecone
    vector = Pinecone.from_documents(splits, embed_model, index_name=index_name)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    st.session_state['documents'] = splits  # Store splits for future reference

# Initialize Chat Model
def initialize_chat_model():
    if google_api_key:
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=google_api_key)
    return None

chat_model = initialize_chat_model()

# Define retrieval and prompt templates
def get_retrieval_context(query):
    """Fetch relevant documents from Pinecone and format them as context."""
    relevant_docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in relevant_docs)

# Prompt template for Q&A
prompt_str = """
Answer the user question based only on the following context:
{context}

Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Function to generate response using Chat Model
def get_response(user_question):
    context = get_retrieval_context(user_question)
    prompt = _prompt.format_prompt(context=context, question=user_question)
    return chat_model.invoke(prompt.to_message_dict())

# Display chat history in UI
st.markdown("<h2 style='color: #007acc; text-align: center;'>Chat with Your Documents 💬</h2>", unsafe_allow_html=True)

# Input for user's question
user_question = st.text_input("Your Question:", key="user_input", placeholder="Ask from documents...")

# Generate response if there is a question and model is ready
if user_question and chat_model and st.session_state['documents']:
    response = get_response(user_question)
    st.session_state['history'].append((f"User: {user_question}", f"AI: {response}"))

# Display conversation history
for i, (user_q, ai_resp) in enumerate(reversed(st.session_state['history'])):
    st.write(user_q)
    st.write(ai_resp)

# Reminders if API keys or documents are missing
if not google_api_key:
    st.warning("Please enter your Google Gemini API Key.")
if not pinecone_api_key:
    st.warning("Please enter your Pinecone API Key.")
if not st.session_state['documents']:
    st.warning("Please upload PDF documents to use as context for answering questions.")
