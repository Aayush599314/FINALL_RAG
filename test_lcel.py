import os
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# dummy setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents([Document(page_content="test")], embeddings)
retriever = vector_store.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="dummy", max_retries=0)

prompt = PromptTemplate.from_template("Context: {context}\nQuestion: {question}")
try:
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    print("Success with 'question'")
except Exception as e:
    print("Error:", e)

prompt2 = PromptTemplate.from_template("Context: {context}\nQuestion: {input}")
try:
    combine_docs_chain = create_stuff_documents_chain(llm, prompt2)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    print("Success with 'input'")
except Exception as e:
    print("Error2:", e)
