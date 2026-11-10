# ===================================
# Part-1 : Importing all the necessary libraries and dependencies
# ===================================
import os
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ===================================
# Part-2 : Loading the env file and initializing the api keys and endpoint url
# ===================================

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
if not qdrant_url:
    raise ValueError("Qdrant Cluster Endpoint Error!")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
if not qdrant_api_key:
    raise ValueError("Qdrant API Error!")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Groq API Error!")

# ===================================
# Part-3 : Connecting to Qdrant Client
# ===================================

qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False, check_compatibility=False)
print("Connected to Qdrant Client Successfully!")

# ===================================
# Part-4 : Initializing the Collection and its details
# ===================================

collection_name = "knowledge_base"
embedding_size = 384

if qdrant_client.collection_exists(collection_name):
    print(f"{collection_name} collection already exists! Hence Deleting it.")
    qdrant_client.delete_collection(collection_name)

qdrant_client.create_collection(collection_name = collection_name, vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE))
print("Collection created!")

# ===================================
# Part-5 : Loading our knowledge base from .txt file
# ===================================

file = open("knowledge_base.txt" , "r", encoding="utf-8")
raw_lines = file.readlines()
file.close()

document = []

for i in raw_lines:
    slash_n_lines = i.strip()
    if slash_n_lines:
        document.append(slash_n_lines)

print(f"Loaded {len(document)} lines of data from knowledge_base.")

# ===================================
# Part-6 : Creating Embedding of loaded knowledge base (document array)
# ===================================

print("Starting to embed the knowledge base loaded")
model = SentenceTransformer("all-MiniLM-L6-v2") #Embedding size = 384
embedding = model.encode(document)
print(f"Generated total {len(embedding)} embeddings.")
print(f"Size of a single embedding : {len(embedding[0])}")

# ===================================
# Part-7 : Creating Points in Qdrant for the embeddings formed
# ===================================

points=[]

for i in range (len(embedding)):
    point = PointStruct(
        id = i+1,
        vector = embedding[i].tolist(),
        payload = {"text" : document[i]}
    )
    points.append(point)

# ===================================
# Part-8 : Upsert the points formed to qdrant
# ===================================

qdrant_client.upsert(collection_name=collection_name, points=points)
print(f"Upserted {len(points)} documents to Qdrant!")

# ===================================
# Part-9 : Search function
# ===================================

def search(query,top_k=3):
    query_vector = model.encode(query).tolist()

    result = qdrant_client.query_points(collection_name=collection_name, query=query_vector, limit=top_k, with_payload=True).points

    return result

# ===================================
# Part-10 : Testing our search function
# ===================================

query = "Today is Sunday. So much rush on roads because people have their plans. At what time Mr. Nadeem should leave office in order to reach his home anyhow before 8pm?"
received = search(query, top_k=3)
for x in received:
    print(f"Score: {x.score:.3f}")
    print(x.payload["text"])
    print()

# ============================================================
# PART 11 — CONNECT TO GROQ
# ============================================================

groq_client = Groq(api_key=groq_api_key)
groq_model = "openai/gpt-oss-120b"


# ============================================================
# PART 12 — ASK LLM FUNCTION
# ============================================================

def ask_llm(question, context):

    msg_sys = f"Answer the question using only the information provided below. If the answer is not in the context, say: I don't know based on the provided information. Context: {context}"

    message_system = {
        "role": "system",
        "content": msg_sys
    }

    message_user = {
        "role": "user",
        "content": question
    }

    messages = [message_system, message_user]

    response = groq_client.chat.completions.create(model=groq_model, messages=messages)
    answer = response.choices[0].message.content
    return answer


# ============================================================
# PART 13 — FULL RAG PIPELINE
# ============================================================

question = "Today is Sunday. So much rush on roads because people have their plans. At what time Mr. Nadeem should leave office in order to reach his home anyhow before 8pm?"

results = search(question, top_k=3)

context = ""
for result in results:
    context = context + result.payload["text"] + "\n"

answer = ask_llm(question, context)

print("\nFinal Answer:")
print(answer)