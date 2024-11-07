from utils import get_graph

graph = get_graph()

# Initial query to count movies
query_count_movies = "MATCH (m1:Movie) WITH count(m1) as count_movies RETURN count_movies"

# Function to add new users (currently not implemented)
def add_new_users():
    print("Not yet")

# Queries for counting users, movies, and ratings
query_no_users = (
    'MATCH (u1:User) '
    'WITH count(u1) as count_users '
    'RETURN count_users'
)

query_no_movies = (
    'MATCH (m1:Movie) '
    'WITH count(m1) as count_movies '
    'RETURN count_movies'
)

query_no_ratings_by_user = (
    'MATCH (u1:User)-[:RATED]->(m1:Movie) '
    'WITH u1, count(m1) AS number_rated_movies '
    'RETURN sum(number_rated_movies) as total_ratings'
)

# Function to execute queries
def get_data_query(query):
    # Use graph.run() to execute the query
    result = graph.run(query)
    return result.data() 

def recomend_by_similar_ratings(user_id, threshold, rec_number=10):
    query = (
        # Similarity normalization: count the number of movies seen by u1
        'MATCH (m1:Movie)<-[:RATED]-(u1:User {id: $user_id}) '
        'WITH count(m1) AS countm, u1 '
        # Recommendation: Retrieve all users u2 who share at least one movie with u1
        'MATCH (u2:User)-[r2:RATED]->(m1:Movie)<-[r1:RATED]-(u1) '
        # Check if the ratings given by u1 and u2 differ by less than 1
        'WHERE (NOT u2 = u1) AND (abs(r2.rating - r1.rating) <= 1) '
        # Compute similarity
        'WITH u1, u2, count(DISTINCT m1) AS rated_count, countm '
        'WITH u1, u2, tofloat(rated_count) / countm AS sim '
        # Keep users u2 whose similarity with u1 is above some threshold
        'WHERE sim > $threshold '
        # Retrieve movies m that were rated by at least one similar user, but not by u1
        'MATCH (m:Movie)<-[r:RATED]-(u2) '
        'WHERE (NOT (m)<-[:RATED]-(u1)) '
        # Compute score and return the list of suggestions ordered by score
        'WITH DISTINCT m, count(r) AS n_u, tofloat(sum(r.rating)) AS sum_r '
        'WHERE n_u > 1 '
        'RETURN m, sum_r / n_u AS score ORDER BY score DESC LIMIT $rec_number'
    )
    
    # Execute the query with parameters
    result = graph.run(query, user_id=user_id, threshold=threshold, rec_number=rec_number)
    return result.data()  # Get the result as a list of dictionaries


result = recomend_by_similar_ratings(13, 0.5)
print(result)

no_users = get_data_query(query_no_users)
print(no_users)

no_movies = get_data_query(query_no_movies)
print(no_movies)

no_queries_by_user = get_data_query(query_no_ratings_by_user)
print(no_queries_by_user)

"""
MATCH (m1:Movie)<-[:RATED]-(u1:User {id: 139})
WITH u1, count(m1) AS countm  // Keep u1 and countm together

// Find users u2 who have rated the same movies as u1
MATCH (u2:User)-[r2:RATED]->(m1:Movie)<-[r1:RATED]-(u1)
WHERE NOT u2 = u1 AND abs(r2.rating - r1.rating) <= 1

// Calculate similarity based on the number of distinct movies rated
WITH u1, u2, count(DISTINCT m1) AS shared_movies_count, countm
WITH u1, u2, tofloat(shared_movies_count) / countm AS sim  // Calculate similarity

// Filter based on the similarity threshold
WHERE sim > 0.5

// Retrieve movies rated by similar users that u1 has not rated
MATCH (m:Movie)<-[r:RATED]-(u2)
WHERE NOT (m)<-[:RATED]-(u1)

// Aggregate ratings for the recommended movies
WITH DISTINCT m, count(r) AS n_u, tofloat(sum(r.rating)) AS sum_r
WHERE n_u > 1  // Ensure that at least one similar user rated the movie

//Return movies and their average rating score, ordered by score
RETURN m, sum_r / n_u AS score
ORDER BY score DESC

"""