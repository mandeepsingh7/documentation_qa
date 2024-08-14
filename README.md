# NVIDIA Cuda Documentation QA

## Project Overview

The "NVIDIA Cuda Documentation QA" project is designed to facilitate users in querying and retrieving information from the NVIDIA Cuda documentation. The project involves creating a custom scraper to extract relevant content from various NVIDIA domains and building a QA system using a combination of web scraping, data storage, and a chatbot interface.

## Features

- **Web Scraping:** Extraction of web pages from NVIDIA Cuda Documentation using Scrapy.
- **Metadata Extraction:** Collection of metadata such as titles, headings (h1 to h6) ,and URLs of web pages.
- **Data Storage:** Saving the extracted data into a JSON file.
- **Database Creation:**
  - **Document Creation:** After scraping the web pages, the content is stored in a JSON file (`cuda_docs.json`). Each entry in the JSON file represents a web page with its content and metadata (such as title, URL, and headings).
  
  - **Chunking the Documents:** The content of each document is often too large to be processed as a single chunk. Therefore, the content is split into smaller, manageable chunks using a text splitter (`RecursiveCharacterTextSplitter`). The splitting is done in such a way that chunks overlap slightly to preserve context between them. In this project, the chunk size is set to 1000 characters with an overlap of 500 characters.
  
  - **Embedding Generation:** Each chunk is then converted into a dense vector representation (embedding) using the OpenAI Embeddings model. Embeddings are numerical representations that capture the semantic meaning of text, making it possible to compare and search for similar chunks.
  
  - **Storing in Chroma Database:** The embeddings along with the associated metadata (like title, URL, and headings) are stored in a Chroma database. Chroma is a vector database that allows for efficient storage and retrieval of embeddings, enabling quick search and comparison of text data.

- **Query and Response System:**
  - **Query Embedding:** When a user submits a query, the text is first converted into an embedding using the same OpenAI Embeddings model. This allows the query to be represented in the same vector space as the document chunks stored in the database.
  
  - **Retrieving Relevant Chunks:** The query embedding is then used to search the Chroma database for the most relevant document chunks. This is done using a similarity search, where the query embedding is compared to the embeddings of the document chunks stored in the database. The database returns the chunks that are most similar to the query, based on their vector representations.
  
  - **Generating Response:** The retrieved chunks are then passed to an OpenAI model (e.g., GPT-3.5) along with a prompt template. The model uses the context provided by these chunks to generate a response to the user's query. This response is then displayed to the user in the Streamlit interface, along with the titles and URLs of the source documents.

- **User Interface:** 
  - **Streamlit Interface:** The `app.py` file creates a user interface using Streamlit. Users can input their questions, and the system will display the answer along with the source titles and URLs. The interface is designed to be user-friendly, allowing seamless interaction with the QA system.

## Project Structure

- `chroma/`: Directory for storing Chroma database files.
- `cuda_scraper/`: Scrapy project directory.
- `data/`: Contains the `cuda_docs.json` file with extracted content.
- `.env`: Environment variables file which contains `OPENAI_API_KEY`.
- `app.py`: Streamlit application for the QA system.
- `create_db.py`: Script to create the Chroma database and save document embeddings.
- `global_settings.py`: Global configuration file with project settings.
- `query.py`: Contains logic to query the Chroma database and generate responses.
- `README.md`: Project documentation.
- `requirements.txt`: Dependencies for the project.

## Implementation Details

### 1. Web Scraping and Metadata Extraction

The project starts by scraping relevant pages from the NVIDIA CUDA Documentation website. The `Scrapy` framework is used for this purpose. The spider is configured to limit its crawl to the following domains:

```python
allowed_domains = ["docs.nvidia.com","nvidia.github.io", "nvlabs.github.io", "developer.nvidia.com"]
```

A depth limit of 5 is set to control how deep the spider will crawl into the website.

During the crawl, the spider extracts the following data from each page:
- **Title**: The title of the webpage.
- **URL**: The URL of the webpage.
- **Headings**: Headings from h1 to h6 to be used later as metadata.
- **Content**: The full body text of the webpage.

This data is then stored in a JSON file named `cuda_docs.json`.

### 2. Database Creation
The `create_db.py` script processes the scraped documents and stores their embeddings in a Chroma database. The main steps include:

- **Loading Documents**: Reading the JSON file and extracting the metadata and content.
- **Text Splitting**: Dividing the content into chunks for better query performance.
- **Storing Embeddings**: Using OpenAI embeddings to represent each chunk and saving them in the Chroma database.
### 3. Query and Response System
The `query.py` script handles user queries by:

- **Fetching Relevant Chunks**: Using the query text to find the most relevant chunks in the Chroma database.
- **Generating Response**: Using a prompt template and the OpenAI model to generate a response based on the retrieved chunks.
- **Returning Source Information**: The system also returns the titles and URLs of the sources from which the answer was derived.
### 4. User Interface
The `app.py` file creates a user interface using `Streamlit`. Users can input their questions, and the system will display the answer along with the source titles and URLs.

### 5. Global Settings
The `global_settings.py` file contains configuration values such as:

- CHUNK_SIZE: 1000
- CHUNK_OVERLAP: 500
- PROMPT_TEMPLATE: The template used for generating responses.
- OPENAI_MODEL: "gpt-3.5-turbo-0125"

## Usage                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                              
- **Run the Web Scraper**: Use Scrapy to extract the required data from the NVIDIA Cuda Documentation.                                                                                                                                                                                                                                                                                                                                                                                                                        
  - cd into `cuda_scraoer` and run the following command:
    ```sh
    scrapy crawl cuda_docs -O ../data/cuda_docs.json
                                                                                                                                                                                                                                                                                                                                                                                                                                             
- **Create the Database**: Execute `create_db.py` to store document embeddings in Chroma.
- **Start the Application**: Run `app.py` to launch the Streamlit interface and start asking questions.

## Dependencies
Make sure to install the required dependencies using:
```sh
pip install -r requirements.txt
```

## Conclusion
This project offers a robust framework for building a documentation QA system. It leverages modern web scraping techniques, an efficient database for document embeddings, and a user-friendly interface for querying the documentation.