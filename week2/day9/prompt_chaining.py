import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api error")

client = Groq(api_key = my_api_key)
model = "openai/gpt-oss-120b"

JD = """
We are hiring for an AI/ML Engineer.

Experience required : 2 years of experience.
Skills : Python, TensorFlow, PyTorch, AWS, Matplotlib, Seaborn, Plotly, Git.
"""

RESUME = """
Candidate name : Md Nadeem Ansari
Experience : 2 years of experience.
Skills : Python, AWS, Matplotlib, Seaborn, Plotly, Git.
"""

def run_llm(system_prompt, user_prompt):
    msg_sys = {
        "role" : "system",
        "content" : system_prompt
    }

    user_msg = {
        "role" : "user",
        "content" : user_prompt
    }

    messages = [msg_sys, user_msg]

    response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    answer = response.choices[0].message.content
    print(answer)

def step1_res_extract(RESUME):
    print("STEP 1")
    system_prompt = f"""
    You are a profession HR assistant. You have to extract the candidate's experience and skill information from the candidate's resume. Do not add guess or invent something from your side. strictly follow the given resume.
    Output format: keep the details and values with comma-separation.
    """
    user_prompt = f"""
    Extract the candidate's information from the resume given.
    {RESUME}
    """

    return run_llm(system_prompt,user_prompt)

def step2_jd_extract(JD):
    print("STEP 2")
    system_prompt = f"""
    You are a profession HR assistant. You have to extract the requirement details about experience and skill from the job description. Do not add guess or invent something from your side. strictly follow the given job description.
    Output format: keep the details and values with comma-separation.
    """
    user_prompt = f"""
    Extract the information from the job description given.
    {JD}
    """

    return run_llm(system_prompt,user_prompt)

def step3_matching(RESUME, JD):
    print("STEP 3")
    system_prompt = f"""
    You are a profession HR assistant. You have to match the candidate's profile (experience and skill information) from the candidate's resume with the requirements for the role from the job description. Do not add guess or invent something from your side. strictly follow the given resume and job description snd generate a matching score.
    """
    user_prompt = f"""
    Match the job description and candidate's resume. Generate a score based on the skills and experience match, in order to check whether the candidate is fit for the role.
    resume : {RESUME}
    job description : {JD}
    """

    return run_llm(system_prompt,user_prompt)

step1_res_extract(RESUME)
sleep(2)
step2_jd_extract(JD)
sleep(2)
step3_matching(RESUME, JD)