from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

def create_workflow(projects):
    class GraphConfig(TypedDict):
        project_name: Literal[tuple(projects)]
        internet_search: bool
    
    workflow = StateGraph(OverallState, config_schema=GraphConfig)
    
    # Add nodes
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("llm_answer", llm_answer)
    workflow.add_node("split_questions", split_question_list)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("transform_docs", transform_docs)
    workflow.add_node("rag_answer", rag_answer)
    workflow.add_node("cite_sources", cite_sources)
    workflow.add_node("sql_agent", sql_agent)
    
    # Set entry point
    workflow.set_entry_point("detect_intent")
    
    # Add edges
    workflow.add_conditional_edges(
        "detect_intent",
        decide_answering_path,
        {
            "greeting": "llm_answer",
            "specific_question": "split_questions",
            "metadata_query": "sql_agent",
            "follow_up_question": "llm_answer"
        }
    )
    
    workflow.add_edge("llm_answer", END)
    workflow.add_edge("sql_agent", END)
    workflow.add_edge("split_questions", "retrieve")
    workflow.add_edge("retrieve", "transform_docs")
    workflow.add_edge("transform_docs", "rag_answer")
    workflow.add_edge("transform_docs", "cite_sources")
    workflow.add_edge("rag_answer", END)
    workflow.add_edge("cite_sources", END)
    
    workflow.set_finish_point("rag_answer")
    
    return workflow.compile() 

def decide_answering_path(state: OverallState) -> str:
    """Decide which path to take based on the detected intent."""
    intent = state.get("intent", "")
    
    if intent == IntentEnum.GREETING:
        return "greeting"
    elif intent == IntentEnum.SPECIFIC_QUESTION:
        return "specific_question"
    elif intent == IntentEnum.METADATA_QUERY:
        return "metadata_query"
    elif intent == IntentEnum.FOLLOW_UP_QUESTION:
        return "follow_up_question"
    else:
        return "specific_question"  # default path 