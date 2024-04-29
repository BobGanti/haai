import os
import sys
import openai
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
INSTRUCTIONS = os.getenv('INSTRUCTIONS') 
ASSISTANT_PROFILE = os.getenv('MD_PROFILE')


# DocumentChunk class to wrap text chunks
class DocumentChunk:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Sets up Streamlit UI components.
def setup_streamlit_ui():
    st.set_page_config(page_title="PA", page_icon="👽")
    st.title("Healthcare Assistant 👽")
    with st.sidebar:
        st.header("Menu")
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        if uploaded_files:
            st.session_state['pdf_files'] = uploaded_files

# Load PDF files from the default directory.
def load_sys_pdfs():
    SYS_PDF_DIR = os.path.join(os.path.dirname(__file__), 'sys')
    SYS_PDFs = []
    for filename in os.listdir(SYS_PDF_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(SYS_PDF_DIR, filename)
            SYS_PDFs.append(file_path)
    return SYS_PDFs

# Extracts text from an uploaded PDF file.
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += (page.extract_text() or '') + '\n'
    return text

# Splits text into chunks without cutting words in half.
def chunk_text(text, chunk_size=2500):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    while text:
        split_point = text.rfind(' ', 0, chunk_size) + 1
        if not split_point:  # No spaces found, hard split at chunk_size
            split_point = chunk_size
        chunks.append(text[:split_point].strip())
        text = text[split_point:]
    return chunks

# Creates a vector store from an uploaded PDF file.
def create_vector_store_from_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    doc_chunks = chunk_text(text)
    doc_objs = [DocumentChunk(chunk) for chunk in doc_chunks]
    vectordb = Chroma.from_documents(
        documents=doc_objs, 
        embedding=OpenAIEmbeddings(), 
        persist_directory=os.path.join(os.path.dirname(__file__), 'sys/data/vectorstores')
    )
    return vectordb
    vectordb.persist()

# Processes uploaded PDF files.
def process_pdfs(pdf_files):
    vectorstores = [create_vector_store_from_pdf(pdf) for pdf in pdf_files]

    st.session_state['vector_stores'] = vectorstores

#Creates a retriever chain for context retrieval.
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", INSTRUCTIONS)
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

# Creates a chain for conversational RAG processing.
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{ASSISTANT_PROFILE}:\n\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Generates a response for the user query using the conversation chain.
def get_response(user_query):
    if 'vector_stores' in st.session_state:
        responses = []
        for vector_store in st.session_state['vector_stores']:
            retriever_chain = get_context_retriever_chain(vector_store)
            conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
            response = conversation_rag_chain.invoke({
                        'chat_history': st.session_state.get('chat_history', []),
                        'input': user_query
                    })
            responses.append(response['answer'])
        return ' '.join(responses)
    return 'Unable to process the query without a valid vector store.'

# Displays conversation history and input box for new queries.
def display_conversation():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [] #  Initialising the chat history

    user_query = st.chat_input("Say something...", key="user_query")
    if user_query:
        response = get_response(user_query)
        st.session_state['chat_history'].extend([
            HumanMessage(content=user_query),
            AIMessage(content=response)
        ])

    for message in st.session_state['chat_history']:
        if isinstance(message, AIMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.write(message.content)

def main():

    setup_streamlit_ui()

    pdf_files = st.session_state.get('pdf_files', [])
    if pdf_files and st.sidebar.button("Process PDFs"):
        with st.sidebar:
            with st.spinner("Processing..."):
                process_pdfs(pdf_files)
                st.success("Processing complete.")
    else:
        pdf_files = load_sys_pdfs()
        process_pdfs(pdf_files) 

    display_conversation()

if __name__ == "__main__":
    main()
