# Resume-Candidate-
!pip install faiss-cpu
!pip install pypdf
!pip install langchain-community
!pip install langchain-openai
!pip install openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "Your API KEY"

# Load candidate resumes or documents
loader = PyPDFLoader('Path of Candidate's Resume.pdf')  # Replace with the actual path to the candidate's resume
documents = loader.load()

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Create embeddings for the documents
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(docs, embeddings)

# Get a query from the user regarding the candidate
query = input("What would you like to know about the candidate? ")
relevant_docs = vectordb.similarity_search(query, k=3)  # Retrieve the top 3 relevant documents
document_texts = "\n\n".join([doc.page_content for doc in relevant_docs])

# Create a prompt for the LLM
prompt = f"Based on the following candidate documents, answer the query: {query}\n\n{document_texts}"

# Initialize the OpenAI model
llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])
response = llm.invoke(prompt)

# Print the response from the LLM
print(response)
