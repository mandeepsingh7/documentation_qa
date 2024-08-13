from global_settings import JSON_PATH, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PATH
import openai
import json
import os
import shutil
from dotenv import load_dotenv
from typing import List
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def main():
    docs = load_docs(JSON_PATH)
    chunks = create_chunks(docs)
    save_embeddings_to_db(chunks)

def load_docs(json_file_path : str):
    docs = []
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    for data_dict in data:
        url = data_dict['url']
        title = data_dict['title']
        h1 = ' '.join(data_dict['h1'])
        h2 = ' '.join(data_dict['h2'])
        h3 = ' '.join(data_dict['h3'])
        h4 = ' '.join(data_dict['h4'])
        h5 = ' '.join(data_dict['h5'])
        h6 = ' '.join(data_dict['h6'])
        content = data_dict['content']
        metadata = {
            'url': url,
            'title': title,
            'h1': h1,
            'h2': h2,
            'h3': h3,
            'h4': h4,
            'h5': h5,
            'h6': h6
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def create_chunks(docs : List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def save_embeddings_to_db(chunks : List[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

if __name__ == '__main__':
    main()