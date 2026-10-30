# 1. import statements and setting up Groq
import os
from pydantic import BaseModel
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep
import re

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key = my_api_key)
model = "openai/gpt-oss-120b"

def get_product_price(product):
    if product=="iPhone 17":
        return 76000
    elif product=="iPhone 16":
        return 65000
    else:
        return 0

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc error!"

tools = {
    "get_product_price" : get_product_price,
    "calculator" : calculator
}

system_prompt = """
You are an shopping assistant.

You have these tools:

get_product_price(product)
calculator(expression)

IMPORTANT:
Call these tools exactly like these examples:
Action : get_product_price("iPhone 17")
Action : calculator("80000-76000")

Never write:
get_product_price(product="iPhone 17")

Never write:
calculator(expression="80000-76000")

Follow these rules:
1. Decide what you have to do next.
2. For that call ONE TOOL at a time.
3. After writing an action, stop immediately.
4. NEVER GUESS OR INVENT TOOLS on your own.
5. Wait until you receive an observation.
6. Then decide your next action and use steps 1-5 recursively as required.
7. When the whole task is complete, give the Final Answer.

Format:

Thought : what you need to do
Action : tool_name(argument)

When finished:

Final Answer : your answer
"""

def run_agent(question):
    messages = [
        {
            "role" : "system",
            "content" : system_prompt
        },
        {
            "role" : "user",
            "content" : question
        }
    ]

    for step in range(5):
        print("\n---------------------")
        print("STEP",step + 1)
        print("----------------------")
        
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)

        answer = response.choices[0].message.content
        print(answer)

        if "Final Answer" in answer:
            break

        match = re.search(r"Action\s*:\s*(\w+)\((.*?)\)", answer)

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')

            if tool_name in tools:
                tool = tools[tool_name]
                observation = tool(tool_input)
            else:
                observation = "tool not found!"

            print("Observation: ", observation)



            messages.append({
                "role" : "assistant",
                "content" : answer
            })

            messages.append({
                "role" : "user",
                "content" : "Observation: "+str(observation)
            })

            sleep(5)


prompt = """
I have 80000 rupees. What is the price of iPhone 17? and how much money will I have left after buying it?
"""

run_agent(prompt)

