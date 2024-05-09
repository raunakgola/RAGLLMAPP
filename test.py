import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Cassandra
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import  create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# with st.sidebar:
#     upload_file = st.file_uploader("upload your file for query")
# if (upload_file is not None):
loader1 = PyPDFLoader("SE_unit31_Copy_Copy.pdf")
pdf_document = loader1.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
document = text_splitter.split_documents(pdf_document)
db1 = Cassandra.from_documents(document,OllamaEmbeddings())
# db = FAISS.from_documents(document,OllamaEmbeddings())

llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context.
Think step by step before providing a detailed answer.
Explain me thinking i am a 12th standard student using example
    
<context>
{context}
<context>
Question: {input}""")

document_chain = create_stuff_documents_chain(llm,prompt)

retriever = db1.as_retriever()

retrieval_chain = create_retrieval_chain(retriever,document_chain)
st.title("RAG LLM APP 🤖📄")
st.markdown("App which can retrieve the data from document and by asking query generate output")
query = st.text_input("write something here")
if st.button("submit"):
    response = retrieval_chain.invoke({"input": f"{query}"})
    st.write(response["answer"])
    # if (response is not None):
    #     st.write_stream(response["answer"])
    # print(response["answer"])
