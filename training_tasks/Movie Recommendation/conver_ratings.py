import csv

# Function to generate Cypher query for Rating relationships
def generate_rating_cypher(user_id, movie_id, rating, timestamp):
    return f"MERGE (u:User {{ id: {user_id} }});\n" \
           f"MERGE (m:Movie {{ id: {movie_id} }});\n" \
           f"MERGE (u)-[r:RATED {{ rating: {rating}, timestamp: {timestamp} }}]->(m);\n"

# File path to your ratings CSV file
csv_file = "ratings.csv"

# Open and read the CSV file
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    # Open a file to store the Cypher queries
    with open("ratings.cypher", "w", encoding='utf-8') as rating_file:
        
        # Loop through each row of the CSV
        for row in reader:
            user_id = row['userId']
            movie_id = row['movieId']
            rating = row['rating']
            timestamp = row['timestamp']
            
            # Generate the Cypher query for the rating relationship
            rating_cypher = generate_rating_cypher(user_id, movie_id, rating, timestamp)
            rating_file.write(rating_cypher)  # Write to ratings.cypher
