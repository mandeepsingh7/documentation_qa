import chromadb
import os 
from src.schemas import MultiQueryOutput 
from src.global_settings import CHROMA_PATH, EMBEDDING_FUNCTION,\
      BM25_PATH, LLM_MODEL, MULTI_QUERY_PROMPT, NUM_QUERIES, CHROMA_K, \
      RRF_K, PROMPT_TEMPLATE, RERANKED_INPUT_K, RERANKED_OUTPUT_K, RERANK_MODEL 
import re
import pickle 
import cohere 
from langchain_chroma import Chroma 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def load_retrievers():

    chroma_db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function= EMBEDDING_FUNCTION
    )

    with open(BM25_PATH, "rb") as f: 
        bm25, docs = pickle.load(f)

    return chroma_db, bm25, docs 

def generate_multi_queries(query: str):

    llm = ChatOpenAI(
        model = LLM_MODEL
    )

    structured_llm = llm.with_structured_output(MultiQueryOutput)

    prompt = MULTI_QUERY_PROMPT.format(query=query, num_queries=NUM_QUERIES)

    response = structured_llm.invoke(prompt)

    multi_queries = response.queries 

    multi_queries.append(query)

    return multi_queries 

def dense_search(chroma_db, query:str):
    results = chroma_db.similarity_search(query, k=CHROMA_K)
    return results

def bm25_search(bm25, docs, query:str):
    tokenized_query = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CHROMA_K]
    return [docs[i] for i in top_n]

def reciprocal_rank_fusion(results_list, k=RRF_K):
    fused_scores = {}
    doc_map = {}
    for results in results_list :
        for rank, doc in enumerate(results):
            chunk_id = doc.metadata['chunk_id']
            if chunk_id not in fused_scores :
                fused_scores[chunk_id] = 0 
                doc_map[chunk_id] = doc 
            fused_scores[chunk_id] += 1 / (k + rank + 1) 

    sorted_ids = sorted (
        fused_scores, 
        key = fused_scores.get,
        reverse=True 
    )
    return [doc_map[chunk_id] for chunk_id in sorted_ids]

def rerank_documents(query: str, docs):
    co = cohere.Client(os.environ['COHERE_API_KEY'])
    texts = [doc.page_content for doc in docs]
    response = co.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=texts,
        top_n=RERANKED_OUTPUT_K
    )
    reranked_docs = [docs[result.index] for result in response.results]
    return reranked_docs

def hybrid_search(query: str, chroma_db, bm25, docs): 
    multi_queries = generate_multi_queries(query)

    dense_results = []
    sparse_results = []

    for q in multi_queries:
        dense_results.extend(dense_search(chroma_db, q)) 
        sparse_results.extend(bm25_search(bm25, docs, q))

    fused_docs = reciprocal_rank_fusion([dense_results, sparse_results])
    final_docs = fused_docs[:RERANKED_INPUT_K]
    reranked_docs = rerank_documents(query, final_docs)
    return reranked_docs

def get_relevant_chunks(query: str):
    chroma_db, bm25, docs = load_retrievers() 
    final_docs = hybrid_search(query, chroma_db, bm25, docs)
    return final_docs

def get_response(query: str, docs):
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(context=context, query=query)

    llm = ChatOpenAI(model=LLM_MODEL)
    response = llm.invoke(prompt)
    return response

def get_final_response(query: str):
    docs = get_relevant_chunks(query) 
    response = get_response(query, docs)
    urls = [doc.metadata.get('url', None) for doc in docs]
    return {
        'response' : response.content,
        'urls': urls
    }