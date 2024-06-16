# Import statements (unchanged)
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Function to read documents from a directory (unchanged)
def readDoc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    docs = file_loader.load()
    return docs

# Example directory for loading documents (modify as needed)
doc_directory = 'docs/'

# Function to chunk documents (unchanged)
def chunkData(docs, chunk_size=500, chunk_overlap=50):
    text_spiltter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_chunks = text_spiltter.split_documents(docs)
    return doc_chunks

# Reading documents
docs = readDoc(doc_directory)
documents = chunkData(docs)

# Initialize HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings()

# Assuming `index` initialization for Pinecone is correct (adjust as per your setup)
os.environ['PINECONE_API_KEY'] = "7625c1cf-dee4-4560-a4b2-71413b619cbb"
index_name = 'quiz-app'
index = PineconeVectorStore.from_documents(documents, index_name=index_name, embedding=embedding)

# Initialize HuggingFaceEndpoint for LLM
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm_hug = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.9, token="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")

# Load QA chain
chain = load_qa_chain(llm_hug, chain_type="stuff")

# Function to retrieve answers
def retrieveAnswer(query):
    doc_search = retrivequery(query)  # assuming `retrivequery` function is defined correctly elsewhere
    response = chain.run(input_documents=doc_search, question=query)
    return response

# Streamlit app setup
st.set_page_config(page_title="Question Generation App")
st.header("Question Generation")

# Streamlit interface for user input
input_topic = st.text_input("Enter Topic:")
difficulty_level = st.selectbox("Difficulty Level:", ('Easy', 'Medium', 'Hard'))
subject = st.selectbox('Select Subject:', ('Python', 'LLM', 'Data Science'))
num_questions = st.number_input('Number of Questions:', value=None)

# Handling user submission
submit_button = st.button("Ask")
if submit_button:
    question = f"Give me {int(num_questions)} MCQ questions of {difficulty_level} difficulty on {input_topic} based on {subject} subject."
    answer = retrieveAnswer(question)
    st.write(answer)
