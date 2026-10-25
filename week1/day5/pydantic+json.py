import os
from pydantic import BaseModel
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key = my_api_key)
model = "openai/gpt-oss-20b"

class Ticket(BaseModel):
    name : str
    email : str
    issue : str

schema = Ticket.model_json_schema()

response_format = {"type" : "json_object"}

system_prompt = f"""
Extract personal information form ticket strictly based on this schema and give a JSON output.
{schema}
"""

text = f"""
Hi! Myself Abdul. I have bought an iPhone Duo smartphone from your store 3 days ago. It isn't turning on. Please reach out to me as soon as possible on my email : abdul@gmail.com
"""

prompt = f"""
This is a customer ticket. Please extract the personal information from this and categorise the issues mentioned by the customer in either of the one : 1. Hardware , 2. Software. For anything else that doesnt match the given categories put it into this category : Others.
{text}
"""

message_system = {
    "role" : "system",
    "content" : system_prompt
}

message = {
    "role" : "user",
    "content" : prompt
}

messages = [message_system , message]

response = client.chat.completions.create(model = model, messages = messages , response_format = response_format)

answer = response.choices[0].message.content
# print(answer)

data_file = json.loads(answer)
ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)