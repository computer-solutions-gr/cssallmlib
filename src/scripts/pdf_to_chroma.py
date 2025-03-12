import os
import PyPDF2
from langchain_core.documents import Document
from cssallmlib.vectordb.chroma_db import ChromaManager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


def parse_pdf(file_path):
    logger.info(f"Parsing PDF file: {file_path}")
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    logger.debug(f"Extracted text from {file_path} with length {len(text)}")
    return text


def chunk_text(text, params):
    logger.info("Chunking text")
    # splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitter = RecursiveCharacterTextSplitter(chunk_size=params["chunk_size"], chunk_overlap=params["chunk_overlap"])
    chunks = splitter.split_text(text)
    logger.debug(f"Created {len(chunks)} chunks")
    return chunks


def store_documents_in_chroma(params):
    logger.info(f"Storing documents in Chroma DB at {params['chroma_db_path']}")
    manager = ChromaManager(collection_name="pdf_collection", path=params["chroma_db_path"])
    for filename in os.listdir(params["pdf_directory"]):
        if filename.endswith('.pdf'):
            file_path = os.path.join(params["pdf_directory"], filename)
            logger.info(f"Processing file: {filename}")
            text = parse_pdf(file_path)
            chunks = chunk_text(text, params)
            documents = [Document(page_content=chunk, metadata={"filename": filename}) for chunk in chunks]
            manager.upsert_documents(documents)
            logger.info(f"Stored {len(documents)} documents for {filename}")


if __name__ == "__main__":
    params = {
        "pdf_directory": "corpus",
        "chroma_db_path": "./experiment_db_3",
        "chunk_size": 5000,
        "chunk_overlap": 300
    }
    if not os.path.exists(params["chroma_db_path"]):
        os.makedirs(params["chroma_db_path"])
    logger.info("Starting PDF to Chroma processing")
    store_documents_in_chroma(params)
    logger.info("Completed PDF to Chroma processing") 