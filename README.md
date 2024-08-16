# AWS-Bedrock-AI-Chat-Model
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)

## This chatBot is built using:
  1. Amazon Bedrock models
  2. LangChain 
  3. Python 
  4. Streamlit 
  5. Boto3 

### Amazon Bedrock is a fully managed serverless service that enables developers to build and deploy chatbot applications with ease.

## Models used:
1. Amazon Titan Embedding G1 - Text to create embeddings 
2. Llama3 to access LLM

### Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web applications.  
  
## Configuration setting of the repository:
1. Install the packages mentioned in **requirements.txt** using command `py -m pip install requirements.txt`
2. Visual Studio Code extensions required-
   1. AWS Toolkit
   2. Python
   3. Python Debugger
   4. WSL
3. Run files using `py -m streamlit run admin.py` and `py -m streamlit run user.py`

## Functionality behind the ChatBot: 
1. Admin application:
   1. It loads the PDF documents and splits it into chunks.
   2. Once the chunks/tokens are generated, then the embeddings are created using the **amazon-titan-embed-text** embedding Model.
   3. Then these embeddings (vector representation of the chunks) are stored as `vector_store` locally using FAISS and stored in 'tmp' folder in your local.
   4. Rename the FAISS and PKL file created in **\tmp** folder to remove `.bin` extension for convenience.

2. User application: 
* It uses the **Streamlit** to give user a web interface where they can chat by maintaining the session state of the user. 
   1. When the application starts, it loads the vector store locally and deseralize the embeddings in it.
   2. Then **LangChain's** **RetrievalQA** does a similarity search between the deserialized embeddings of the `vector_store` and user prompt. 
   3. It retrieve 5 relevant documents as per the user query to build the context.
   4. Then using the Prompt template, it feeds the similarily search and provides the question and context to the LLM to generate user understandable response using **Llama3** Model.
   5. Finally, the response from **LLM** is being displayed to the user on browser in **Streamlit** interface.


