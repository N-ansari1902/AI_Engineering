import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key = my_api_key)
model = "openai/gpt-oss-120b"

prompt = """
Explain internet and its indepth working in about 250 words.
"""
message = {
    "role" : "user",
    "content" : prompt
}

messages = [message]

# response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
# answer = response.choices[0].message.content
# print(answer)
print("Thinking.....\n")
stream = client.chat.completions.create(model=model, messages=messages, temperature=0.0, stream=True)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)