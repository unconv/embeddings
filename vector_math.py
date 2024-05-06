from embeddings import cosine_similarity
import numpy as np

dictionary = np.load("dictionary.npy", allow_pickle=True).item()

vector = dictionary["woof"] - dictionary["dog"] + dictionary["cow"]

results = {}

for word, embedding in dictionary.items():
    results[word] = cosine_similarity(vector, embedding)

results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for word, score in results[:5]:
    print(f"{word}: {score}")
