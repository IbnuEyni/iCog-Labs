import logging
import streamlit as st
import pandas as pd
import numpy as np
from py2neo import Graph
from utils import get_graph

# Constants
K = 20
ALPHA = 0.001
BETA = 0.01
ITERATIONS = 100

logging.basicConfig(level=logging.INFO)

graph = get_graph()

def load_ratings() -> pd.DataFrame:
    """Load user-movie ratings from the database.
    
    Returns:
        pd.DataFrame: DataFrame containing user IDs, movie IDs, movie titles, and ratings.
    """
    query_ratings = ('MATCH (u:`User`)-[r:`RATED`]->(m:`Movie`) '
                     'RETURN u.id AS user_id, m.id AS movie_id, '
                     'm.title AS movie_title, r.rating AS rating')
    user_movie_ratings = graph.run(query_ratings).data()
    
    ratings_df = pd.DataFrame(user_movie_ratings)
    ratings_df['rating'] = pd.to_numeric(ratings_df['rating'])
    return ratings_df

# Matrix Factorization class
class MF:
    def __init__(self, R: np.ndarray, K: int, alpha: float, beta: float, iterations: int):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i + 1) % 20 == 0:
                logging.info(f"Iteration: {i + 1} ; error = {mse:.4f}")

        return training_process

    # Other methods remain unchanged...

# Predict rating for a specific user and movie
def predict_rating(ratings_df: pd.DataFrame, user_id: int, movie_id: int):
    """Predict the rating for a specific user and movie.
    
    Args:
        ratings_df (pd.DataFrame): DataFrame of user ratings.
        user_id (int): ID of the user.
        movie_id (int): ID of the movie.
        
    Returns:
        Tuple[Optional[float], Optional[float], Optional[float]]: Actual rating, predicted rating, and MSE.
    """
    ratings_df_full_pivot = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    user_of_interest_index = ratings_df[(ratings_df['movie_id'] == movie_id) & (ratings_df['user_id'] == user_id)].index

    if user_of_interest_index.empty:
        return None, None, None

    ratings_df_dr = ratings_df.drop(user_of_interest_index)

    ratings_df_pivot = ratings_df_dr.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    R = np.array(ratings_df_pivot)
    mf = MF(R, K=K, alpha=ALPHA, beta=BETA, iterations=ITERATIONS)

    # Train the model
    training_process = mf.train()

    predicted_rating = mf.get_rating(user_id, movie_id)
    return ratings_df_full_pivot.loc[user_id, movie_id], predicted_rating, mf.mse()

# Streamlit UI remains unchanged...
