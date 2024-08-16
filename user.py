import boto3
import streamlit as st

## FAISS is used for vector store
from langchain_community.vectorstores import FAISS 

## prompt and Langchain library
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

## ollama used for creating embeddings and access Large Language Model
#from langchain_community.llms import ollama
#from langchain_community.embeddings import OllamaEmbeddings

## Output parsers 
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough

# Bedrock to create embeddings and access Large Language Model 
from langchain_aws import BedrockLLM
from langchain_community.embeddings import BedrockEmbeddings

st.set_page_config(page_title="Doc AI Bot-Admin", page_icon="🤖")
folder_path = "/tmp/"
file_name = "spring-boot"

#def get_llm():
#   llm = ollama(model="llama3")
#  return llm

# Session to connect to AWS services
session = boto3.Session(profile_name="aarzoo")

# Create a Bedrock Client to connect to Bedrock services
bedrock_client = session.client("bedrock-runtime",region_name="us-east-1")

# Connect to amazon-titan-embed-text embedding model using Bedrock Client 
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock_client
)

# Connect to llama3 LLM using Bedrock Client
def get_llm():
    llm = BedrockLLM(
       model_id="meta.llama3-8b-instruct-v1:0",
       client=bedrock_client,
       model_kwargs={"temperature": 0.0, "top_p": 0.1, "max_gen_len": 512},
    )
    return llm

# Get response from LLM
def get_response(llm, question):

   # Deserialize the already created embeddings in admin.py to have a similarity search with user prompt
    vector_store = FAISS.load_local(
        index_name=file_name,
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization= True
    )

    ## create user prompt 
    prompt = """<|start_header_id|>user<|end_header_id|>
    You're a helpful AI assistant and use only the vector store information.
    If you don't know the answer, just say that you don't know, don't try to make up an answer and don't show in ordered list.
    <|eot_id|>
    <|start_header_id|›user<|end_header_id|›
    What can you help me with? {context}
    Here is the question: {question}
    If user has asked about you or just send hi, then reply about you and not anything from vector store.
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|›"""

    PROMPT = ChatPromptTemplate.from_template(prompt)
    
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

 # Either use RetrievalQA or RAG chain to have similarity search between the deserialized embeddings and user prompt 
    qa=RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever( 
            search_type="similarity",
            search_kwargs={"k": 5} # To use 5 tokens for response
    ),
    return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})

    answer=qa({"context": qa, "query":question})
    return answer['result']

    # rag_chain = (
    #     {
    #         "context": vectorstore.as_retriever(
    #             search_type="similarity", search_kwargs={"k":5}
    #         )
    #         | format_docs,
    #         "question": RunnablePassthrough(),
    #     }
    #     | PROMPT
    #     | llm
    #     | StrOutputParser()
    # )
    # response = rag_chain.invoke(question)
    #return response

def main():
    st.header("PDF AI Chat!")
    st.write("💬 Ready to Chat!")

    #Initialize chat history to append everything to the same session state of the user
    if "messages" not in st.session_state:
      st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask your query"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            llm = get_llm()
            # Feeding the similarity search between vectorstore and prompt on llm model to generate user understandable response
            response = get_response(llm,prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role" : "assistant", "content" : response})

if __name__=="__main__":
    main()


        







