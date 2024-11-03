from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import PydanticToolsParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_cohere.rerank import CohereRerank

def setup_intent_detection():
    prompt = hub.pull("intent_detection")
    llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
    llm_with_tools = llm.bind_tools(tools=[Intent])
    return prompt | llm_with_tools

def setup_question_detection():
    prompt = hub.pull("split_questions")
    llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
    llm_with_tools = llm.bind_tools(tools=[QuestionList])
    return prompt | llm_with_tools

def setup_cite_sources_chain():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    llm_with_tools = llm.bind_tools([CitedSources])
    prompt = hub.pull("cite_sources")
    return prompt | llm_with_tools | PydanticToolsParser(tools=[CitedSources])

def setup_sql_agent_chain(db_engine):
    llm = OpenAI(streaming=True)
    db = SQLDatabase(engine=db_engine)
    prompt_template = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Question: {input}"""
    
    PROMPT = PromptTemplate(
        input_variables=["input", "dialect"], 
        template=prompt_template
    )
    
    return SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT)

def setup_rag_answer_chain():
    llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0)
    prompt_template = hub.pull("answer_question")
    return prompt_template | llm | StrOutputParser() 