from words import words
from embeddings import get_embedding
import numpy as np

dictionary = {}

for word in words:
    embedding = get_embedding(word)
    dictionary[word] = np.array(embedding)

np.save("dictionary.npy", dictionary)
