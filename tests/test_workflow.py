import pytest
from src.graph import create_workflow
from langchain_core.messages import HumanMessage

@pytest.mark.asyncio
async def test_basic_workflow():
    initial_state = {
        "messages": [HumanMessage(content="Hello, how are you?")],
        "documents": [],
        "intent": "",
        "question_list": [],
        "cited_sources": ""
    }
    
    workflow = create_workflow(["test_project"])
    config = {
        "configurable": {
            "project_name": "test_project",
            "internet_search": True
        }
    }
    
    result = await workflow.ainvoke(initial_state, config=config)
    assert "messages" in result
    assert len(result["messages"]) > 1 