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
role = "user"
prompt = "The patient is experiencing blockage in urine and also some kind of pyogens are coming out of the urine. He is not able to urinate properly and feels like his bladder is full. What could be the underlying disease ?"

message_system = {
    "role" : "system",
    "content" : "You are a diagnosis system for a medical institution where you will be given the symptoms as observed in the patient and you have to match them close to facts and then tell the most probable diagnosed ailment or disease that person may have. Give out one word answer."}

message = {
    "role" : role,
    "content" : prompt
}

messages = [message_system, message]

response = client.chat.completions.create(model = model, messages = messages, temperature=1)

# print(response)
# print("######################################################################")

answer = response.choices[0].message.content
print(answer)