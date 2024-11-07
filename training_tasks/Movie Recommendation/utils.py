from py2neo import Graph
import credentials

def get_graph():
    return Graph(credentials.NEO4J_URI, auth=(credentials.NEO4J_USERNAME, credentials.NEO4J_PASSWORD))

def test_connection():
    graph = get_graph()
    try:
        query = "RETURN 1"
        result = graph.run(query).evaluate()  # Use `evaluate()` to get a single value
        if result == 1:
            print("Database connection established successfully.")
        else:
            print("Unexpected response from the database.")
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
test_connection()

