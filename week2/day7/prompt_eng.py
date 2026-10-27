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

def llm_response(text):
    message_system = {
        "role" : "system",
        "content" : "#ROLE: You are a customer support executive in a mobile/laptop company. #TASK: You have to categorise the given customer's complaint into one of the categories specified. #CONSTRAINT: Specified categories are 1. Billing, 2. Return, 3. Technical Issue. #OUTPUT FORMAT: The answer you have to give should be one-word only and from one of the categories specified under constraint. #EXAMPLE: Suppose, if customer's csomplaint is about refund, your answer should be Return. #FALLBACK: If the customer's complaint doesn't fall under any of the categories as specified under constraints, simply answer OTHER for that."
    }

    message = {
        "role" : "user",
        "content" : f"Classify this complaint:\n<text>\n{text}\n</text>"   #always use delimiters when passing the code into the message to prevent prompt injection vulnerability.
    }

    messages = [message_system , message]

    response = client.chat.completions.create(model=model , messages=messages , temperature=0.0)
    answer = response.choices[0].message.content
    return answer  #using return instead of print for better compatibility with production usage.


complaint = f"""
Hi! Myself Abdul. I have bought an iPhone Duo smartphone from your store 3 days ago. It isn't turning on. Please reach out to me as soon as possible on my email : abdul@gmail.com .
"""

print(llm_response(complaint))