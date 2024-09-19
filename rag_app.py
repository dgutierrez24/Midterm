import chainlit as cl
import os
import tiktoken
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Tiktoken length function
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text)
    return len(tokens)

# Function to load and split PDFs
def load_and_split_pdfs_by_paragraphs(directory, chunk_size=500, chunk_overlap=50):
    documents = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        separators=["\n\n", "\n"]
    )
    
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            
            loader = PyMuPDFLoader(file_path)
            pdf_documents = loader.load()
            
            for page_num, page in enumerate(pdf_documents):
                splits = text_splitter.split_text(page.page_content)
                for i, split in enumerate(splits):
                    documents.append(Document(
                        page_content=split,
                        metadata={
                            "filename": filename,
                            "page_number": page_num + 1,
                            "chunk_number": i + 1
                        }
                    ))
    
    return documents

# RAG prompt template
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
"""

@cl.on_chat_start
async def start():
    # Initialize the RAG system
    cl.Message(content="Initializing the RAG system... This may take a moment.").send()
    
    # Set the directory containing your PDF files
    current_directory = os.getcwd()  # Update this path
    
    # Load and split PDFs
    docs = load_and_split_pdfs_by_paragraphs(current_directory)
    
    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create Qdrant vector store
    qdrant_vectorstore = Qdrant.from_documents(
        docs,
        embedding_model,
        location=":memory:",
        collection_name="extending_context_window_llama_3",
    )
    
    # Create retriever
    qdrant_retriever = qdrant_vectorstore.as_retriever()
    
    # Create RAG prompt
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    
    # Initialize OpenAI chat model
    openai_chat_model = ChatOpenAI(model="gpt-4o-mini")
    
    # Create the RAG chain
    rag_chain = (
        {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
        | rag_prompt
        | openai_chat_model
        | StrOutputParser()
    )
    
    # Store the RAG chain in the user session
    cl.user_session.set("rag_chain", rag_chain)
    
    cl.Message(content="RAG system initialized. You can now ask questions about the PDF documents.").send()

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the RAG chain from the user session
    rag_chain = cl.user_session.get("rag_chain")
    
    # Use the RAG chain to process the user's question
    response = rag_chain.invoke({"question": message.content})
    
    # Send the response back to the user
    await cl.Message(content=response).send()

if __name__ == "__main__":
    cl.run()
