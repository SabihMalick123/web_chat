import streamlit as st
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import LlamaIndex
from langchain_community.embeddings import OllamaEmbeddings
import ollama

# Function to load, split, and retrieve documents
def load_and_retrieve_docs(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict() 
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = LlamaIndex.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain
def rag_chain(url, question):
    retriever = load_and_retrieve_docs(url)
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Streamlit interface
def main():
    st.title("RAG Chain Question Answering")
    st.write("Enter a URL and a query to get answers from the RAG chain.")
    
    url = st.text_input("URL")
    question = st.text_input("Question")
    
    if st.button("Get Answer"):
        if url and question:
            with st.spinner("Loading..."):
                answer = rag_chain(url, question)
                st.write("Answer:", answer)

if __name__ == "__main__":
    main()
