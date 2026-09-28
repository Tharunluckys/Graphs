from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage



@tool
def add(a: int, b: int):
    """Add two numbers."""
    return a + b


@tool
def mul(a: int, b: int):
    """Multiply two numbers."""
    return a * b


tools = [add, mul]

 

llm1 = ChatOllama(
    model="llama3.2",
    temperature=0
).bind_tools(tools)

 

evaluator_llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0
)

 

class AgentState(MessagesState):
    evaluation: str


 

def agent(state: AgentState):

    prompt = SystemMessage(
        content="""You are a helpful AI agent.
Use tools when necessary."""
    )

    response = llm1.invoke(
        [prompt] + state["messages"]
    )

    return {
        "messages": [response]
    }



def evaluator(state: AgentState):

    question = state["messages"][0].content
    answer = state["messages"][-1].content

    prompt = f"""
You are an evaluator.

Question:
{question}

Agent answer:
{answer}

Is the agent's answer correct?

Return exactly:
PASS

or

FAIL
"""

    response = evaluator_llm.invoke(prompt)

    return {
        "evaluation": response.content
    }




def eval_decision(state: AgentState):

    evaluation = state["evaluation"].strip().upper()

    if evaluation.startswith("PASS"):
        return "pass"

    return "retry"



graph = StateGraph(AgentState)

graph.add_node("agent", agent)

graph.add_node(
    "tools",
    ToolNode(tools)
)

graph.add_node(
    "evaluator",
    evaluator
)



graph.add_edge(
    START,
    "agent"
)



graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": "evaluator"
    }
)



graph.add_edge(
    "tools",
    "agent"
)


graph.add_conditional_edges(
    "evaluator",
    eval_decision,
    {
        "pass": END,
        "retry": "agent"
    }
)


app = graph.compile()



result = app.invoke({
    "messages": [
        (
            "user",
            "Add 12 + 13 and then multiply the result by 4."
        )
    ]
})

print(result)