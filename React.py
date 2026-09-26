

from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage,SystemMessage
from langchain_ollama import ChatOllama
from typing import TypedDict,Sequence,Annotated

class Agent(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int):
    '''This function returns the sum of two numbers '''
    return a + b
@tool
def mul(a:int,b:int):
    '''This function returns the product of two numbers'''
    return a * b
tools = [add,mul]

llm = ChatOllama(model='llama3.2').bind_tools(tools)

def chatbot(state:Agent):
    prompt = SystemMessage(
        '''You're an AI agent.
        Use the only tools provided to perform math.
        Don't perform math by yourself'''
    )
    response = llm.invoke([prompt]+state['messages'])
    return {'messages':[response]}

def loop(state:Agent):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return 'end'
    else:
        return 'continue'
toolnode = ToolNode(tools)
graph = StateGraph(Agent)
graph.add_node('chatbot',chatbot)
graph.set_entry_point('chatbot')
graph.add_node('tools',toolnode)
graph.add_conditional_edges(
    'chatbot',
    path = loop,
    path_map={'continue':'tools','end':END}
)
graph.add_edge('tools','chatbot')
app = graph.compile()
app

inputs = {'messages':[('user','''Add 12+13 and then multiply the result by 4''')]}
result = app.invoke(inputs)
print(result['messages'][-1].content)


for message in result["messages"]:
    print(type(message).__name__)
    print(message)
    print("------")