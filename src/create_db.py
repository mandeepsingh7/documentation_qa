from pathlib import Path 
from dotenv import load_dotenv 

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

import re
import json 
import os 
import shutil 
import pickle 
from typing import List

from global_settings import JSON_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PATH, EMBEDDING_MODEL, BM25_PATH
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi 

def clean_text(text: str):
    '''Normalize scraped text by removing non breaking spaces.'''
    text = text.replace('\xa0', ' ')
    return text

def load_docs(json_file_path: str):
    '''
    Load scraped json data and convert each page into a Langchain Document with structured metadata.
    '''
    docs = []
    
    with open(json_file_path) as f:
        data = json.load(f)
    
    for data_dict in data:
        url = data_dict['url']
        title = data_dict.get('title', '')

        h2 = clean_text(" ".join(data_dict.get('h2', '')))
        h3 = clean_text(" ".join(data_dict.get('h3', '')))

        content = clean_text(data_dict.get('content', ''))

        # Combine title, headings and content for context 
        page_text = f"""
Title : {title}
H2 : {h2}
H3 : {h3}

{content}
"""
        metadata = {
            'url': url, 
            'title': title,
            'h2': h2,
            'h3': h3
        }

        docs.append(Document(page_content=page_text, metadata=metadata))

    return docs

def create_chunks(docs: List[Document]):
    '''Split documents into overlapping chunks for embedding and retrieval.'''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "]
    )

    chunks = text_splitter.split_documents(docs)

    # Assign unique id to each chunk (used for identification during RRF ranking)
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i

    return chunks

def save_embedding_to_db(chunks: List[Document]):
    '''Generate embeddings and store them in a Chroma vector database.'''

    # We recreate the embeddings each time this file is run to avoid stale embeddings and to avoid duplicate chunks. 
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_PATH
    )

    print("Saved chunks:", db._collection.count())

def build_bm25_index(chunks: List[Document]): 
    '''Build a BM25 index for hybrid retrieval.'''

    tokenized_corpus = [
        re.findall(r"\w+", chunk.page_content.lower())
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)

    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25, chunks), f)

    print("Saved BM25 index")


def main():
    '''Pipeline for creating retrievers'''
    docs = load_docs(JSON_PATH)
    chunks = create_chunks(docs)
    save_embedding_to_db(chunks)
    build_bm25_index(chunks)

if __name__ == "__main__":
    main()
