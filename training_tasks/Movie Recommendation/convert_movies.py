import csv

def generate_movie_cypher(movie_id, title):
    title = process_apostrophe(title)
    return f"CREATE (m:Movie {{ id: {movie_id}, title: '{title}' }});\n"

def generate_genre_cypher(movie_id, genres):
    cypher_query = ""
    genres_list = genres.split("|")
    
    for index, genre in enumerate(genres_list):
        genre_upper = process_apostrophe(genre.upper())
        cypher_query += f"MERGE (g:Genre {{ name: '{genre_upper}' }});\n"
        cypher_query += f"MERGE (m:Movie {{ id: {movie_id} }})-[:HAS_GENRE {{ position: {index + 1} }}]->(g);\n"
    
    return cypher_query

def process_apostrophe(text):
    result = []
    skip_next = False

    for i, char in enumerate(text):
        if skip_next:
            skip_next = False 
            continue
        if char == "'":
            skip_next = True 
        else:
            result.append(char)
    
    return ''.join(result)

csv_file = "movies.csv"

with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    with open("movies.cypher", "w", encoding='utf-8') as movie_file, open("genres.cypher", "w", encoding='utf-8') as genre_file:
        
        for row in reader:
            movie_id = row['movieId']
            title = row['title']
            genres = row['genres']
            
            movie_cypher = generate_movie_cypher(movie_id, title)
            movie_file.write(movie_cypher)  
            
            genre_cypher = generate_genre_cypher(movie_id, genres)
            genre_file.write(genre_cypher) 
