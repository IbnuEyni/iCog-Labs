import streamlit as st
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase

# Connect to Neo4j
graph = Neo4jGraph(
    url=st.secrets['NEO4J_URI'],
    username=st.secrets['NEO4J_USERNAME'],
    password=st.secrets['NEO4J_PASSWORD']
)

# Schema extractor function
@st.cache_data
def get_schema_text(url, user, password):
    driver = GraphDatabase.driver(url, auth=(user, password))
    # Query for node properties
    node_query = """
    MATCH (source)-[relationship]->(target)
    RETURN DISTINCT {
    source: labels(source)[0],
    predicate: type(relationship),
    target: labels(target)[0]
    } AS result
    """
 
    with driver.session() as session:
        results = session.run(node_query).data()

    driver.close()
    
    return results

# Initialize schema
uri = st.secrets['NEO4J_URI']  
user = st.secrets['NEO4J_USERNAME']  
password = st.secrets['NEO4J_PASSWORD']

schema = get_schema_text(uri, user, password)
# Optional: Display the schema in Streamlit
st.write(schema)

