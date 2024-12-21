from bot import get_gemini_response


def extract_entities(question):
    prompt = """
    You are just a converter and an expert in converting English questions into JSON requests for a biological annotation service that retrieves
    data about biological entities (nodes) and their relationships (predicates). The response should only contain two arrays: 
    `nodes` and `predicates`. Additionally, note that the number of objects in predicates will always be one less than the number of objects in nodes.
    
    - **Node**: These are typically biological entities such as genes, proteins, cells, or molecules. Nodes are often nouns or noun phrases in the query.
    - **Property**: These are characteristics or attributes of nodes, such as expression location, size, or any measurable trait. 
        Properties are typically adjectives or numeric values in the query.
    - **Relationship**: These are interactions or connections between nodes, often expressed as verbs or prepositions (e.g., "transcribed_from", "regulates", "binds to", "interacts with").

    For simpler queries, such as retrieving all nodes or edges, structure the JSON to contain only one of these arrays:
    - `nodes` array for listing all entities.
    - `predicates` array for listing all relationships (edges).

    For queries involving only one node type (e.g., all genes or all proteins), structure the JSON to include a single 
    object in the `nodes` array. This object should specify the `type` field with the relevant node type, such as `"gene"`, 
    `"protein"`, `"transcript"`, etc., while making the 'node_id': 'n1' and the `id`: ''  to indicate a request for all nodes of that type.

    For queries where a specific node is requested with a given `id` (e.g., a unique identifier such as an Ensembl or HUGO ID), 
    structure the JSON with a single object in the `nodes` array. This object should include:
    - `node_id` with a unique identifier, such as "n1"
    - `id` to specify the provided node ID  
    - `type` to indicate the type of node, such as "gene", "protein", "transcript", "exon" etc.
    - `properties` dictionary, if additional attributes or properties of this node are requested.
    - Don't include predicates or relationships.

    For queries requesting specific nodes with given properties, structure the JSON with one or more objects in the `nodes` 
    array. Each object should include:
    - `node_id` with a unique identifier, such as "n1", "n2", etc.
    - `id`, "", since we need an annotation query to get it to specify the unique node identifier (e.g., Ensembl or HUGO ID)
    - `type` to indicate the type of node, such as "gene", "protein", "transcript", etc.
    - `properties` dictionary containing the specific attributes or properties requested for each node. 
      This allows you to specify the exact properties that should be returned for each node (e.g., "accessions", "location", "function") but no predicates nor relationships.
    Instruction:
    - Use a `predicates` array with one object specifying:
      - `source`: `node_id` of the first node
      - `target`: `node_id` of the second node
      - `type`: relationship type (e.g., "transcribed_from").
    """
    response = get_gemini_response(question, prompt)
    return response

# Step 2: Classify Entities and Relationships
def classify_entities_and_relationships(entities_response):
    if "predicates" not in entities_response or "predicates" == []:
        return entities_response
    prompt = f"""
    Using the extracted entities and relationships below:
    
    {entities_response}    

    Follow these rules:

    1. **Single Relationship Between Two Nodes**
       - If we have a single relationship and two nodes:
         - Use a `nodes` array with two objects.
           - Each object should include:
             - `node_id`: "n1", "n2"
             - `id`: if known, else ""
             - `type`: node type
             - `properties`: if any specific attributes are requested.
        - Use a `predicates` array with one object specifying:
          - `source`: `node_id` of the first node
          - `target`: `node_id` of the second node
          - `type`: relationship type (e.g., "transcribed_from").

    2. **Indirect Relationship Between Two Nodes**
        - Nodes Array:
          If two nodes do not have a direct relationship:
            Search for an intermediate node that has a direct relationship with both nodes.
            Include all three nodes (Node A, Intermediate Node, and Node B) in the nodes array.
            - Each object should include:
             - `node_id`: "n1", "n2"
             - `id`: if known, else ""
             - `type`: node type
             - `properties`: if any specific attributes are requested.

        - Predicates Array:
          In the case where an intermediate node is used:
              Add two predicates entries:
                First Predicate: Relationship between the first node (Node A) and the intermediate node.
                Second Predicate: Relationship between the intermediate node and the second node (Node B).
              - Use a `predicates` array with one object specifying:
                - `source`: `node_id` of the first node
                - `target`: `node_id` of the second node
                - `type`: relationship type (e.g., "transcribed_from").

    Guidelines:
    - Ensure the JSON output does not include ` ``` ` or the term `json`.
    """

    response = get_gemini_response(entities_response, prompt)
    return response

# Step 3: Construct JSON
def construct_json(classified_data):
    prompt = f"""
    Based on the classification provided below:
    
    {classified_data}
    
    Generally structure the JSON using two main arrays: `nodes` and `predicates`. Ensure that when predicates are included, at least two nodes 
    are specified to establish the relationship. If an additional node ID is not provided, set it to an empty string `""` and include the node 
    in the array. The `nodes` array should contain objects for each entity, where each object includes a mandatory `node_id` 
    (e.g., "n1" for the first node, "n2" for the second node, etc.), an `id` (e.g., Ensembl or HUGO ID), 
    a `type` (e.g., "gene", "transcript", "protein", "enhancer"), and an optional `properties` dictionary for additional attributes.

    The `predicates` array defines relationships, with each object specifying the relationship type and identifying the 
    source and target nodes by their `node_id`.
    the JSON code should not contain the ` ``` ` in beginning or end or the word `json` in the output.
    """
    response = get_gemini_response(classified_data, prompt)
    return response