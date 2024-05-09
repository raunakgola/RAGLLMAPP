from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import  create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
loader1 = PyPDFLoader("method.pdf")
pdf_document = loader1.load()
# loader = TextLoader("")
# text_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
document = text_splitter.split_documents(pdf_document)

db = FAISS.from_documents(document,OllamaEmbeddings())

# query = " "
# result = db.similarity_search(query)
# db.si
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

retriever = db.as_retriever()

retrieval_chain = create_retrieval_chain(retriever,document_chain)

response = retrieval_chain.invoke({"input":"what is this pdf all about?"})
print(response["answer"])
