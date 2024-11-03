import logging
import uuid
from langchain_core.messages import AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)

async def detect_intent(state: OverallState, config):
    messages = state["messages"]
    question = messages[-1].content
    chat_history = messages[:-1]
    
    intent_detection = setup_intent_detection()
    response = await intent_detection.ainvoke({
        "chat_history": chat_history, 
        "question": question
    })
    
    logger.info(f"Intent detection response: {response['intent']}")
    return {"intent": response['intent']}

async def split_question_list(state: OverallState, config):
    split_questions = setup_question_detection()
    question = state['messages'][-1].content
    
    questions = await split_questions.ainvoke({"QUESTION": question})
    logger.info("Question was split into %s parts", len(questions))
    return {"question_list": questions}

async def retrieve(state: OverallState, config):
    question = state['messages'][-1].content
    project_name = config.get('configurable', {}).get('project_name')
    
    # Vector search
    results = await db.asimilarity_search(question=question, project_name=project_name)
    documents = []
    
    if results:
        for doc in results:
            try:
                doc[0].metadata['score'] = doc[1]
                documents.append(doc[0])
            except Exception as e:
                logger.error(f"Error processing document: {e}")
    
    # Filter by threshold
    docs_above_threshold = [
        doc for doc in documents 
        if doc.metadata.get('score', 0) > 0.5
    ]
    
    # Internet search fallback
    internet_search = config.get('configurable', {}).get('internet_search', False)
    if internet_search and len(docs_above_threshold) == 0:
        web_search_tool = TavilySearchResults()
        web_docs = await web_search_tool.ainvoke({"query": question}, max_results=3)
        
        for doc in web_docs:
            document = Document(
                page_content=doc["content"],
                metadata={
                    "type": "search",
                    "url": doc["url"],
                    "uuid": str(uuid.uuid4()),
                    "source": "web_search"
                }
            )
            documents.append(document)
    
    # Rerank results
    reranker = CohereRerank(model="rerank-multilingual-v3.0")
    reranked_results = reranker.rerank(documents, query=question)
    
    filtered_docs = [
        doc for doc in documents 
        if doc.metadata.get('relevance_score', 0) > 0.7
    ]
    
    return {"documents": filtered_docs}

async def transform_docs(state: OverallState, config):
    documents = state["documents"]
    if not documents:
        return {"documents": []}
        
    seen_uuids = set()
    unique_documents = []
    
    for doc in documents:
        uuid = doc.metadata.get('uuid')
        if uuid not in seen_uuids:
            seen_uuids.add(uuid)
            unique_documents.append(doc)
            
    return {"documents": unique_documents}

async def rag_answer(state: OverallState, config):
    question = state["messages"][-1].content
    sources = "\n\n".join(
        f"{i}. {doc.page_content}" 
        for i, doc in enumerate(state["documents"], 1)
    )
    
    chain = setup_rag_answer_chain()
    response = await chain.ainvoke({
        "SOURCES": sources,
        "QUESTION": question
    })
    
    return {"messages": AIMessage(content=response)}

async def cite_sources(state: OverallState, config):
    question = state["messages"][-1].content
    sources = ""
    
    if not state["documents"]:
        logger.error('Unable to answer, no sources found')
        return {"cited_sources": ""}
        
    for i, doc in enumerate(state["documents"], 1):
        source = doc.metadata.get('source', 'Unknown')
        url = doc.metadata.get('url', 'No URL provided')
        sources += f"{i}. {doc.page_content}\n\nURL: {url}\nSOURCE: {source}\n"
    
    chain = setup_cite_sources_chain()
    response = await chain.ainvoke({
        "SOURCES": sources,
        "QUESTION": question
    })
    
    return {"cited_sources": response}

async def sql_agent(state: OverallState, config):
    chain = setup_sql_agent_chain(db.engine)
    
    for _ in range(3):  # Retry logic
        try:
            result = await chain.arun(state["messages"][-1].content)
            return {"messages": AIMessage(content=result)}
        except Exception as e:
            logger.error(f"Error in SQL agent: {e}")
    
    return {"messages": AIMessage(content="Failed to execute SQL query")}

async def llm_answer(state: OverallState, config):
    messages = state["messages"]
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    response = await llm.ainvoke(messages)
    return {"messages": AIMessage(content=response.content)} 