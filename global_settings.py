CHROMA_PATH = "chroma"
JSON_PATH = "data/cuda_docs.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500

PROMPT_TEMPLATE = '''
You are a chatbot which facilitates users about queries on NVIDIA Cuda Documentation.
Your purpose is to use the below documentation for NVIDIA Cuda to answer the subsequent documentation questions.
If the answer cannot be found in the documentation, then write "I could not find an answer.".
Context :
{context}

---

Answer the below question based on the above context:
{query}
'''
CHROMA_K = 5
OPENAI_MODEL = "gpt-3.5-turbo-0125"
MODEL_TEMPERATURE = 0.5
MAX_TOKENS = None
TIMEOUT = None
MAX_RETRIES = 2