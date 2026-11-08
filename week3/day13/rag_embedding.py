import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a,b):
    return np.dot(a,b) / (
        np.linalg.norm(a)*np.linalg.norm(b)
    )

model = SentenceTransformer("all-MiniLM-L6-v2") #384 sized vector
# text = "I love newdelhi store's sweets."

# embedding=model.encode(text)
# print(embedding.shape)
# print(embedding[:10])

t1 = "Rohit's age is 25"
t2 = "Rohit is 25 years old"

v1 = model.encode(t1)
v2 = model.encode(t2)

print(cosine_similarity(v1,v2))