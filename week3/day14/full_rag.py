import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key = my_api_key)
groq_model = "openai/gpt-oss-20b"

model = SentenceTransformer("all-MiniLM-L6-v2") #384 sized vector

knowledge_base = [
    "Rahul works during 9am - 5pm from monday to saturday in his office.",
    "His office is 20km away from his house",
    "It takes around nearly 1-1.5hrs for him to reach his office on heavy traffic days through car. On normal days he reaches before 1hr that is nearly 50mins",
    "On evenings and nights after his work shifts, going back home takes 1hr on normal days with normal traffic.",
    "On days of any occassion, the traffic is so much and it takes him around 2.5hr to travel back to his home from office.",
    "He takes day off on sundays and dont go anywhere and takes rest at home."
]
kb_embedding = model.encode(knowledge_base)

def cosine_similarity(a,b):
    return np.dot(a,b) / (
        np.linalg.norm(a)*np.linalg.norm(b)
    )

def retrieval(q_embedding):
    scores = []
    for i , line in enumerate(kb_embedding):
        values = cosine_similarity(q_embedding , line)
        scores.append((values, knowledge_base[i]))
    scores.sort(reverse=True)
    return scores[0]

def ask_llm(question,context):
    
    msg_sys = f"""Use the given context only to answer the question, no other sources. You may use your thinking for calculations and all, but Do not hallucinate, invent or guess anything. Context : {context}"""
    
    message_system = {
        "role" : "system",
        "content" : msg_sys
    }

    message = {
        "role" : "user",
        "content" : question
    }

    messages = [message_system, message]

    response = client.chat.completions.create(model=groq_model, messages=messages, temperature=0.0)
    answer = response.choices[0].message.content
    return answer

query = "Today is Sunday. So much rush on roads because people have their plans. At what time Rahul should leave office in order to reach his home anyhow before 8pm?"
q_embedding = model.encode(query)
match,context = retrieval(q_embedding)
# print(match)
# print(context)
result = ask_llm(query,context)
print(result)


