import streamlit as st
import langchain
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
load_dotenv()


#Read the document

def readDoc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    docs = file_loader.load()
    return docs

doc = readDoc('docs/')


# Divide docs into chunks

def chunkData(docs,chunk_size=500,chunk_overlap=50):
    text_spiltter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc = text_spiltter.split_documents(docs)

    return doc

documents = chunkData(docs=doc)

# Embedding technique of Hugging face
embedding = HuggingFaceEmbeddings()

# Search in Vector DB -pinecone
from pinecone import Pinecone
pc = Pinecone(api_key="7625c1cf-dee4-4560-a4b2-71413b619cbb")


os.environ['PINECONE_API_KEY'] = "7625c1cf-dee4-4560-a4b2-71413b619cbb"
index_name = 'quiz-app'
print("Start...")
index = PineconeVectorStore.from_documents(doc,index_name="quiz-app",embedding=embedding)
print("End...")
# Retrive result

def retrivequery(query,k=2):
    matching = index.similarity_search(query=query,k=k)
    return matching



os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm_hug = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=128, temperature=0.9, token="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"
)

chain = load_qa_chain(llm_hug,chain_type="stuff")

# Search answers from Vector DB

def retriveAnswer(query):
    doc_search = retrivequery(query)
    response = chain.run(input_documents = doc_search,question = query)
    return response


st.set_page_config(page_title="Question Generation App")
st.header("Question Generation")

input = st.text_input("Enter Topic:", key="input")

col1, col2, col3 = st.columns(3)

with col1:
    level = st.selectbox("Dificulty Level",('Easy', 'Medium', 'Hard'), index=0)

with col2:
    topic = st.selectbox('Select Subject', ('Python', 'LLM', 'Data Science'), index=0)

with col3:
    questionNumber = st.number_input('Number of Question',value=None)


submit = st.button("Ask")
if submit:
    question = f"Give me {int(questionNumber)} MCQ questions of {level} difficulty on {input} based on {topic} subject."
    print(question)
    # answer = retriveAnswer(question)
    # st.write(answer)
