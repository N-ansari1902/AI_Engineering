import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key = my_api_key)
model = "openai/gpt-oss-20b"

def retrieval_software(question):
    if "pratyush" in question:
        if "age" in question:
            return {"age" : 25}
        elif "net worth" in question:
            return {"net worth" : 2000}
        else:
            return None
    else:
        return None

def ask_llm(question):
    context = retrieval_software(question)
    
    msg_sys = f"""answer in one line only. Use the given context only to answer the question, no other sources. Do not hallucinate, invent or guess anything. Context : {context}"""
    
    message_system = {
        "role" : "system",
        "content" : msg_sys
    }

    message = {
        "role" : "user",
        "content" : question
    }

    messages = [message_system, message]

    response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    answer = response.choices[0].message.content
    return answer

question = "what is pratyush's net worth?"
print(ask_llm(question))