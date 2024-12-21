from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain_core.prompts import PromptTemplate
from utils import get_session_id
from neo4j.exceptions import CypherSyntaxError
from llm import llm
from graph import graph
from tools.cypher import cypher_qa, entity_extract

# Create a movie chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a expert providing information about knwoledges from graph data base you are connected with."),
        ("human", "{input}"),
    ]
)

movie_chat = chat_prompt | llm | StrOutputParser()

# Validate and execute Cypher query
def validate_cypher_query(cypher_query):
    try:
        result = graph.query(cypher_query)
        return True, result
    except CypherSyntaxError as e:
        return False, f"Invalid Cypher Query: {e}"

def cypher_qa_with_validation(input_text):
    cypher_query = cypher_qa(input_text)
    is_valid, response = validate_cypher_query(cypher_query)
    return response if is_valid else f"Error: {response}"

# Create a set of tools
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=movie_chat.invoke,
    ),
    # Tool.from_function(
    #     name="Entity Extraction",
    #     description = "Extract entities from the user queries",
    #     func=entity_extract
    # ),
    Tool.from_function(   
        name="Information",
        description="Provide information about the content in the neo4j graph you are connected with using Cypher",
        func=cypher_qa
    )
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent prompt template
agent_prompt = PromptTemplate.from_template(""" 
Be as helpful as possible and return as much information as possible.
You are a biocypherKG expert providing information about bioscience.

you are a Graph Search tool equipped to provide insightful answers by delving into, comprehending, \
and condensing information from Graph Database.
TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```
Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
# Create the agent
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
def generate_response(user_input):
    """ 
    Handler to call the conversational agent and return a response to be rendered in the UI.
    """
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}}, 
    )
    return response['output'] if response else "Error: Unable to generate response."
