import streamlit as st
from llm import llm
from graph import graph

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from neo4j.exceptions import CypherSyntaxError
# Function to validate Cypher query
def validate_cypher_query(cypher_query):
    try:
        # Attempt to run the query on the graph
        result = graph.run(cypher_query)
        return True, result.data()  # If successful, return True with the result data
    except CypherSyntaxError as e:
        # If there's a syntax error or execution failure, return False with the error message
        return False, str(e)

# Cypher generation template
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT
)

# Entity extraction template
_DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """Extract all entities from the following text. As a guideline, a proper noun is generally capitalized. You should definitely extract all names and places.

Return the output as a single comma-separated list, or NONE if there is nothing of note to return.

EXAMPLE
i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.
Output: Langchain
END OF EXAMPLE

EXAMPLE
i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff. I'm working with Sam.
Output: Langchain, Sam
END OF EXAMPLE

Begin!

{input}
Output:"""
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["input"], template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
)
entity_extract = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=ENTITY_EXTRACTION_PROMPT
)

# Validate the generated Cypher query
is_valid, response = validate_cypher_query()
if is_valid:
    st.write("Valid Cypher Query:", response)
else:
    st.error("Invalid Cypher Query:", response) 
