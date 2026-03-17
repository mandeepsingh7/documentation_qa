from pydantic import BaseModel, Field 
from typing import List 

class MultiQueryOutput(BaseModel):
    '''Schema for structured LLM output used in multi query retrieval.'''
    queries: List[str] = Field(description="A list of alternative search queries generated for the original query.")