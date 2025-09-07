import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#  Load the dataset
df = pd.read_csv("C:/Users/USER/Downloads/movies.csv")
print("Sample of the dataset:")
print(df.head())

# Create a User-Item Matrix
user_item_matrix = df.pivot_table(index="user_id", columns="movie_title", values="rating ")
print("\nUser-Item Matrix:")
print(user_item_matrix)

# Fill NaN with 0 and compute cosine similarity
matrix_filled = user_item_matrix.fillna(0)
similarity = cosine_similarity(matrix_filled)
print("\nUser Similarity Matrix:")
print(similarity)

# Recommend movies for a specific user 
user_index = 0  # corresponds to user_id = 1
similar_users = similarity[user_index]

# Find the most similar user (excluding self)
most_similar_user_index = np.argsort(similar_users)[-2]
recommended_movies = user_item_matrix.iloc[most_similar_user_index].dropna().index.tolist()

# Get movies rated by the target user
rated_by_user = user_item_matrix.iloc[user_index].dropna().index

# Recommend only unseen movies
recommended = [movie for movie in recommended_movies if movie not in rated_by_user]
print(f" Final recommendations for User 1: {recommended}")


