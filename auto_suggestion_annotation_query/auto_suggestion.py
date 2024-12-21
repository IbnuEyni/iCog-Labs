import os
import Levenshtein
from dotenv import load_dotenv
from fuzzywuzzy import process
from neo4j import GraphDatabase

load_dotenv()

class NodeSearcher:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        )

    def close(self):
        self.driver.close()

    def get_property_values(self, label, property_name):
        """
        Retrieves all values of the specified property for nodes with the given label.
        """
        with self.driver.session() as session:
            cypher_query = f"MATCH (n:{label}) WHERE n.{property_name} IS NOT NULL RETURN n.{property_name} AS prop_value"
            result = session.run(cypher_query)
            return set([record["prop_value"] for record in result])

    def suggest_similar_values(self, label, property_name, input_value):
        """
        Suggests the most similar values to the input_value from the database using both
        fuzzy matching and Levenshtein distance and returns a set of unique values with
        their similarity percentage.
        """
        existing_values = self.get_property_values(label, property_name)
        if not existing_values:
            return "No existing values found for the given label and property."

        # Step 1: Use FuzzyWuzzy to get initial matches
        fuzzy_matches = process.extract(input_value, existing_values, limit=10)

        # Step 2: Calculate Levenshtein distance for these matches and store with scores
        similarity_results = []
        for match, fuzzy_score in fuzzy_matches:
            levenshtein_distance = Levenshtein.distance(input_value, match)
            # Calculate a normalized Levenshtein similarity score (optional)
            levenshtein_similarity = max(0, 100 - levenshtein_distance * 5)  # Scale as needed

            # Combine FuzzyWuzzy and Levenshtein scores
            if fuzzy_score >= 70 or levenshtein_distance <= 3:
                final_score = max(fuzzy_score, levenshtein_similarity)
                similarity_results.append((match, final_score))

        # Return a set of unique results with their similarity percentage
        return sorted(similarity_results, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    node_searcher = NodeSearcher()

    try:
        label = input("Enter node label: ")
        property_name = input("Enter property name: ")
        input_value = input(f"Enter value to search for similar {property_name}: ")

        suggestions = node_searcher.suggest_similar_values(label, property_name, input_value)
        
        if isinstance(suggestions, str):
            print(suggestions)
        else:
            print("Similar values found with their similarity percentages:")
            for value, score in suggestions:
                print(f"Value: {value}") 
                print(f"Similarity Score: {score:.2f}%")
            
    finally:
        node_searcher.close()
