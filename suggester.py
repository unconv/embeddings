from embeddings import get_embedding, cosine_similarity
import numpy as np

movie_embeddings = np.load("movie_embeddings.npy", allow_pickle=True).item()

while True:
    prompt = input("What kind of movie do you want to see?\n")
    prompt_embedding = get_embedding(prompt)

    results = {}

    for movie, movie_embedding in movie_embeddings.items():
        results[movie] = cosine_similarity(prompt_embedding, movie_embedding)

    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for movie, score in results[:2]:
        print(f"{movie}: {score}")
