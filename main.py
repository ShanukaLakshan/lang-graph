import asyncio
from src.db import DB
from src.graph import create_workflow
from langchain_core.messages import HumanMessage

async def main():
    # Initialize database
    db = DB()
    
    # Get list of projects (implement this in your DB class)
    projects = db.list_projects()
    
    # Create the workflow
    workflow = create_workflow(projects)
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content="What can you tell me about project X?")],
        "documents": [],
        "intent": "",
        "question_list": [],
        "cited_sources": ""
    }
    
    # Run the workflow
    config = {
        "configurable": {
            "project_name": projects[0],  # Use first project as example
            "internet_search": True
        }
    }
    
    result = await workflow.ainvoke(initial_state, config=config)
    
    # Print the result
    print("Final Answer:", result["messages"][-1].content)
    if result.get("cited_sources"):
        print("\nSources:", result["cited_sources"])

if __name__ == "__main__":
    asyncio.run(main()) 