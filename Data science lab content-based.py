import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and clean the dataset
df = pd.read_csv("C:/Users/USER/Downloads/movies with genres.csv")

df.columns = df.columns.str.strip()
df["movie_title"] = df["movie_title"].str.strip()
df["genre"] = df["genre"].fillna("Unknown").str.strip()

#Create a TF-IDF matrix for genres
unique_movies = df.drop_duplicates("movie_title").reset_index(drop=True)
tfidf = TfidfVectorizer(stop_words=None)
genre_matrix = tfidf.fit_transform(unique_movies["genre"])

#Compute cosine similarity between movies
genre_similarity = cosine_similarity(genre_matrix)

#Map movie titles to indices
movie_indices = pd.Series(unique_movies.index, index=unique_movies["movie_title"])

#Define recommendation function
def recommend_by_genre(user_id, top_n=5):
    #Get movies rated highly by the user
    user_rated = df[(df["user_id"] == user_id) & (df["rating"] >= 4)]
    liked_titles = user_rated["movie_title"].str.strip().tolist()
    liked_indices = [movie_indices[title] for title in liked_titles if title in movie_indices]

    #Aggregate similarity scores
    scores = genre_similarity[liked_indices].sum(axis=0)

    #Exclude already rated movies
    rated_titles = set(liked_titles)
    recommendations = [
        unique_movies["movie_title"][i]
        for i in scores.argsort()[::-1]
        if unique_movies["movie_title"][i] not in rated_titles
    ]

    return recommendations[:top_n]

user_id = 1
recommended = recommend_by_genre(user_id)
print(f" Genre-based recommendations for User {user_id}:")
for title in recommended:
    print("-", title)
