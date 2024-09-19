import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

from langchain.document_loaders import PyMuPDFLoader

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)

len(split_chunks)

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tiktoken

# # Initialize the tiktoken tokenizer for the specific model
# encoding = tiktoken.get_encoding("gpt-3.5-turbo")

# # Define the length function based on token count
# length_function = lambda text: len(encoding.encode(text))

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(
        text,
    )
    return len(tokens)

def load_and_split_pdfs_by_paragraphs(directory, chunk_size=500, chunk_overlap=50):
    documents = []
    
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function= tiktoken_len,
        #length_function=len,
        separators=["\n\n", "\n"]
    )
    
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            
            # Load the PDF
            loader = PyMuPDFLoader(file_path)
            pdf_documents = loader.load()
            
            # Split each page into paragraphs
            for page_num, page in enumerate(pdf_documents):
                splits = text_splitter.split_text(page.page_content)
                for i, split in enumerate(splits):
                    documents.append({
                        "filename": filename,
                        "page_number": page_num + 1,
                        "chunk_number":  i+1,
                        "content": split
                    })
    
    return documents

# Get the current directory
current_directory = "/Users/danielgutierrez/Midterm/pdfs"
#current_directory = os.getcwd()
print(f"Current directory: {current_directory}")

# List files in the current directory
print("Files in the current directory:")
print(os.listdir())

# Load and split PDFs from the current directory
docs = load_and_split_pdfs_by_paragraphs(current_directory)

# Print information about the loaded documents
# print(f"\nNumber of chunks created: {len(docs)}")
# print("\nSample of loaded chunks:")
# for i, doc in enumerate(docs[:5]):  # Print first 5 chunks as a sample
#     print(f"File: {doc['filename']}, Page {doc['page_number']}, Chunk {doc['chunk_number']}")
#     print(f"Content preview: {doc['content'][:3000]}...\n")

# print(f"Total chunks: {len(docs)}")

from langchain_openai.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

from langchain_community.vectorstores import Qdrant

qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="extending_context_window_llama_3",
)

qdrant_retriever = qdrant_vectorstore.as_retriever()


from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

from langchain_openai import ChatOpenAI

openai_chat_model = ChatOpenAI(model="gpt-4o-mini")

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

rag_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | rag_prompt | openai_chat_model | StrOutputParser()
    
)

rag_chain.invoke({"question" : "What is the main goal of the NIST AI Risk Management Framework (AI RMF)?"})