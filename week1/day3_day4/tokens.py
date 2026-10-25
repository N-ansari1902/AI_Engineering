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
prompt1 = "Hi!"
prompt2 = "Who is Dr. APJ Abdul Kalam ?"
prompt3 = "Write an essay on the topic 'Land and People' in about 100 words."

prompts = [prompt1, prompt2, prompt3]

for i in prompts:
    message = {
        "role" : role,
        "content" : i
    }

    messages = [message]
    response = client.chat.completions.create(model = model, messages = messages, max_tokens=350)
    # answer = response.choices[0].message.content
    # print(answer)
    usage = response.usage

    print(f"Prompt= {i} : Prompt Tokens= {usage.prompt_tokens} ; Response Tokens= {usage.completion_tokens} ; Total Tokens used= {usage.total_tokens} ; Finish Reason= {response.choices[0].finish_reason}")