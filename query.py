from global_settings import CHROMA_PATH, PROMPT_TEMPLATE, CHROMA_K, OPENAI_MODEL, MODEL_TEMPERATURE, MAX_TOKENS, MAX_RETRIES, TIMEOUT
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Tuple
from langchain.docstore.document import Document


def get_relevant_chunks(query_text: str):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=CHROMA_K)
    return results


def get_response(query_text: str, results: List[Tuple[Document, float]]):
    context_text = '\n\n---\n\n '.join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query_text)
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=MODEL_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES
    )
    response = llm.invoke(prompt)
    return response


def get_final_response(query_text: str):
    results = get_relevant_chunks(query_text)
    response = get_response(query_text, results)
    titles = [doc.metadata.get('title', None) for doc, _score in results]
    urls = [doc.metadata.get('url', None) for doc, _score in results]
    return {
        'response_content': response.content,
        'source_titles': set(titles),
        'source_urls': set(urls)
    }


if __name__ == '__main__':
    question = 'How to install Cuda?'
    response_dict = get_final_response(question)
    print(f'Response : {response_dict["response_content"]}\nSource Titles : {response_dict["source_titles"]}\nSource '
          f'URLs : {response_dict["source_urls"]}')