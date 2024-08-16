import boto3
import streamlit as st

## Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader 
from langchain_community.document_loaders import PyPDFLoader

## FAISS 
from langchain_community.vectorstores import FAISS

## Ollama embeddings 
#from langchain_community.embeddings import OllamaEmbeddings

# bedrock embeddings 
from langchain_community.embeddings import BedrockEmbeddings

st.set_page_config(page_title="PDF AI Bot-admin",page_icon="🤖")
folder_path = "/tmp/"

# Session to connect to AWS services
session = boto3.Session(profile_name="aarzoo")

# Create a Bedrock Client to connect to Bedrock services
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")

# Connect to amazon-titan-embed-text embedding model using Bedrock Client 
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock_client
)

# Load the PDF document and split the document into chunks
def load_split_file():
    with st.spinner("Loading pdf..."):
        loader = PyPDFLoader("./spring-boot.pdf")
        documents = loader.load()
        st.success("PDF loaded")

    with st.spinner("Creating tokens..."):    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        st.success(texts[1])
        return texts
    
## create embeddings from chunks using Bedrock model to be stored in vector DB
def create_vector_store(tokens):
    with st.spinner("Create embeddings..."):
        vectorstore_faiss = FAISS.from_documents(
            tokens,
            embedding=bedrock_embeddings
        )
        st.success(vectorstore_faiss.index.ntotal)

    file_name = "spring-boot.bin"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
    st.success("Embeddings created")
    return True

def main():
    texts = load_split_file()
    create_vector_store(texts)
    st.success("Done!!")

if __name__ == "__main__":
    main()

    




