from enum import Enum
from pydantic import BaseModel, Field
from typing import List, TypedDict, Literal, Sequence
from langchain_core.messages import BaseMessage

class IntentEnum(str, Enum):
    GREETING = "greeting"
    SPECIFIC_QUESTION = "specific_question"
    METADATA_QUERY = "metadata_query"
    FOLLOW_UP_QUESTION = "follow_up_question"

class Intent(BaseModel):
    intent: IntentEnum

class CitedSources(BaseModel):
    source: str = Field(description="The source of the information")
    url: str = Field(description="The URL associated with the source")
    source_type: str = Field(description="The type of source")

class QuestionList(BaseModel):
    questions: List[str] = Field(description="List of individual questions")

class OverallState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    intent: str
    question_list: List[str]
    cited_sources: str 