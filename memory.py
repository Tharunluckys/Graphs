
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model='llama3.2',
    temperature=0,
    stream=True
)

def chatbot(state: MessagesState) -> MessagesState:
    print('AI : ', end='', flush=True)
    response = ''
    for chunk in llm.stream(state['messages']):
        print(chunk.content, end='', flush=True)
        response += chunk.content
    print()
    # Add complete AI response to conversation
    state['messages'].append(AIMessage(content=response))
    return state


# Setup Graph
graph = StateGraph(MessagesState)
graph.add_node('chatbot', chatbot)
graph.add_edge(START, 'chatbot')
graph.add_edge('chatbot', END)
app = graph.compile()


# Conversation history
history = []
query = input('You : ')
while query.strip().lower() != '/bye':
    history.append(HumanMessage(content=query))  # -> Add user's message to history 
    result = app.invoke({'messages': history})
    history = result['messages'] # Update history with graph's returned messages
    query = input('You : ')