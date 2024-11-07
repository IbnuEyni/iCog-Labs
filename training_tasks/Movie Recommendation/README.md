# Movie Recommendation and Rating prediction App

## Overview
The Movie Recommendation App is a web application built with Streamlit that allows users to predict movie ratings and receive movie recommendations based on user preferences. The app utilizes matrix factorization techniques to provide accurate predictions and recommendations, making it an effective tool for movie enthusiasts.

### How It Works
1. **Data Retrieval**: The app connects to a Neo4j database to retrieve user and movie data. Neo4j stores data in nodes and relationships, allowing for efficient querying and data manipulation using the Cypher query language.

2. **Matrix Factorization**: The app employs a matrix factorization algorithm to analyze user ratings and predict the ratings for unrated movies. It optimizes the user and movie feature matrices to minimize prediction errors.

3. **User Interaction**: Users can input their user ID and movie ID to predict ratings or simply input their user ID to receive movie recommendations based on their previous ratings.

## Neo4j and Cypher
- **Neo4j**: A graph database that allows you to store and query data in a graph structure, making it suitable for applications that involve complex relationships, such as social networks and recommendation systems.

- **Cypher**: The query language for Neo4j, which allows you to create, read, update, and delete data in the graph database. Cypher syntax is designed to be intuitive and resembles ASCII art, making it easy to visualize queries.

## Features
- **Predict Movie Rating**: Enter a user ID and a movie ID to get the predicted rating for that movie.
- **Get Recommendations**: Enter a user ID to receive personalized movie recommendations based on their previous ratings.

## Technologies Used
- Python
- Streamlit
- Pandas
- Py2neo (for Neo4j database interaction)
- Matrix Factorization algorithm

## Installation
1. **Clone the repository**:
   git clone https://github.com/IbnuEyni/neo4j-movie-recommendation.git
   cd neo4j-movie-recommendation
2. **Create a virtual environment (optional but recommended):**:
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install the required packages**:
    pip install -r requirements.txt
4. **Set up Neo4j**: Make sure you have a Neo4j database running. Update the credentials.py file with your Neo4j database credentials.

## Usage

1. **Run the Streamlit app**:

    streamlit run app.py

2. Open your web browser and navigate to http://localhost:8501.

3. Choose between "Predict Rating" and "Get Recommendations" from the sidebar.

4. Enter the necessary user ID and movie ID as prompted, and click the respective button to see the results.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.