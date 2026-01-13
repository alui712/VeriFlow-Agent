import os
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- STATE DEFINITION ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question : str
    generation : str
    documents : List[str]

# --- NODES ---

def retrieve(state):
    """
    Retrieve documents from Tavily Search
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Search Tool
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": question})
    
    # Extract content
    web_results = "\n".join([d["content"] for d in docs])
    
    return {"documents": [web_results], "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Keep the answer concise.
        
        Question: {question} 
        Context: {context} 
        Answer:"""
    )
    
    chain = prompt | llm
    generation = chain.invoke({"context": documents, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation.content}

def transform_query(state):
    """
    Refine the query if the answer was not grounded (Self-Correction)
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert at optimizing database queries.
        The previous search for the question '{question}' did not yield a valid answer.
        Look at the previous question and re-write it to be a better search query.
        Output only the updated query string."""
    )
    
    chain = prompt | llm
    better_question = chain.invoke({"question": question})
    
    return {"question": better_question.content}

# --- EDGES (LOGIC) ---

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'yes' means that the answer is supported by the facts."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    grader_chain = grade_prompt | structured_llm_grader
    score = grader_chain.invoke({"generation": generation, "documents": documents})

    if score.binary_score == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED, TRYING AGAIN---")
        return "not supported"

# --- GRAPH BUILD ---
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build the entry point
workflow.set_entry_point("retrieve")

# Add edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("transform_query", "retrieve")

# Add Conditional Edge
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "not supported": "transform_query"
    }
)

# Compile
app = workflow.compile()

# --- EXECUTION ---
if __name__ == "__main__":
    from pprint import pprint
    
    print("Hello! I am your VeriFlow Agent. Type 'quit' to exit.")
    
    while True:
        # Get user input
        user_question = input("\nUser: ")
        
        # Check for exit condition
        if user_question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
            
        # Run the workflow
        inputs = {"question": user_question}
        
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished Node: {key}")
        
        # Print final answer
        print("\n--- FINAL ANSWER ---")
        print(value["generation"])