


from langgraph.graph import StateGraph,START,END,MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
 


llm = ChatOllama(model='llama3.2',temperature=0,stream=True)

def chatbot(state:MessagesState)->MessagesState:
    print(' AI : ',end='',flush=True)
    response = ''
    for i in llm.stream(state['messages']):
        print(i.content,end='',flush=True)
        response += i.content
    print()
    return {'messages':AIMessage(response)}

graph = StateGraph(MessagesState)
graph.add_node('chatbot',chatbot)
graph.add_edge(START,'chatbot')
graph.add_edge('chatbot',END)
app = graph.compile()

user = input(' you  : ')
while user.strip().lower() != '/bye':
    result = app.invoke({'messages':['user',user]})
    print(result)
    user = input(' you : ')