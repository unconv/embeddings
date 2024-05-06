from movies import movies
from embeddings import get_embedding
import numpy as np

movie_embeddings = {}

for movie, description in movies.items():
    embedding = get_embedding(description)
    movie_embeddings[movie] = embedding

np.save("movie_embeddings.npy", movie_embeddings)
