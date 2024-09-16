from neo4j import GraphDatabase

# Connect to Neo4j
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "NEO$Jneo"))

# Function to run a query and fetch all records
def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        # Collect the result in a list 
        records = [record for record in result]
        return records

# Example Cypher query
skill_query = """
MATCH (amir:Person {name: "Amir Ahmedin"})-[:HAS_SKILL]->(skill:Skill)
RETURN skill.name
"""

# Query for education
education_query = """
MATCH (amir)-[:STUDIED_AT]->(edu:Education)
RETURN edu.name, edu.course, edu.completion_date
"""

# Query for experience
experience_query = """
MATCH (amir)-[:WORKED_AT]->(work:EXPERIENCE)
RETURN work.company, work.position, work.duration, work.loaction
"""

# Run the query and collect the result
skill_result = run_query(skill_query)
education_result = run_query(education_query)
experience_result = run_query(experience_query)
print(len(education_result))

# Print the skill results
print("Skills:")
for record in skill_result:
    print(f"- {record['skill.name']}")

print("\nEducation:")
for record in education_result:
    print(f"- {record['edu.name']} ,  {record['edu.course']} ,  {record['edu.completion_date']}")

print("\nExperience:")
for record in experience_result:
    print(f"- {record['work.company']} ,  {record['work.position']} ,  {record['work.duration']} ,  {record['work.loaction']}")

# Close the driver connection
driver.close()
