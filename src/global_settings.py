from pathlib import Path 
from langchain_openai import OpenAIEmbeddings 

BASE_DIR = Path(__file__).resolve().parent.parent 

CHROMA_PATH = BASE_DIR / "storage" / "chroma"
BM25_PATH = BASE_DIR / "storage" / "bm25" / "bm25_index.pkl"
JSON_PATH = BASE_DIR / "data/cuda_runtime_docs_13_03_2026.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_FUNCTION = OpenAIEmbeddings(model=EMBEDDING_MODEL)

NUM_QUERIES = 4
CHROMA_K = 20 
RRF_K = 60 

RERANKED_INPUT_K = 40 
RERANKED_OUTPUT_K = 5 

RERANK_MODEL = "rerank-v4.0-fast"

MULTI_QUERY_PROMPT = '''
You are generating search queries to retrieve technical documentation.
Generate {num_queries} diverse search queries that help retrieve different relevant sections of the documentation.

IMPORTANT RULES:
- Do NOT change technical identifiers (function names, API names, class names).
- If the query contains technical identifiers (camelCase, snake_case, or code terms), they MUST remain unchanged in every query.
- Preserve the original technical meaning.


DIVERSITY REQUIREMENT:
The queries should target different documentation aspects such as:
- definition or description
- usage or behavior
- related concepts
- error cases 
You can target other stuff if you feel necessary.

Avoid simple paraphrases. Each query should help retrieve different relevant documentation chunks.

Original query: {query}

Return one query per line.
'''

PROMPT_TEMPLATE = '''
You are an assistant that answers questions using NVIDIA CUDA Runtime documentation.

Use ONLY the information provided in the context to answer the question.

Rules:
- Do not use prior knowledge about CUDA.
- If the answer is not present in the context, say: "I could not find an answer in the provided documentation."
- Do not make up functions, parameters, or behavior.
- Prefer quoting exact terminology from the documentation.

Context :
{context}

---

Question:
{query}

Answer:
'''

CHROMA_K = 5
LLM_MODEL = "gpt-4.1-mini"
MODEL_TEMPERATURE = 0.5
MAX_TOKENS = None
TIMEOUT = None
MAX_RETRIES = 2