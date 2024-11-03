import openai
from openai import OpenAI
import os
from typing import Optional
import pydantic
import json
import re


user_prompt_correct = """
What is 10 * 173?

step 1: 
10 * 170 = 1700

step 2: 
10 * 3 = 30

step 3:
1700 + 30 = 1730
"""

user_prompt_incorrect = """
What is 10 * 173?

step 1: 
10 * 170 = 1700

step 2: 
10 * 3 = 20

step 3:
1700 + 20 = 1720
"""

class MathResponse(pydantic.BaseModel):
  correct: bool
  error_step: Optional[int]
  how_to_fix: Optional[str]

# results takes in a list of dictionaries (ex: sample_output.json from zihan's part 1 section)
# returns another json object of dictionaries following response_format specified above
def get_corrections(results: list[dict], client: openai.OpenAI, verbose: bool = False) -> list[dict]:
  corrections = []
  for result in results:
    string_representation = "\n".join(result["steps"])
    correction_string = check_math_answer(string_representation, client)
    if verbose:
       print(correction_string)
    correction_json = json.loads(correction_string)
    corrections.append(correction_json)
  return corrections

def check_math_answer(prompt: str, client: openai.OpenAI) -> MathResponse:
  completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are are helping a student solve a math problem. Don't give away the answer, but instead guide them. They will provide you with step by step reasoning for solving a problem, and if they made a mistake, it's your job to identify the first step where they made a mistake, and provide a hint to help them correct it. Do not outright give them the answer. If they are correct, return error_step and how_to_fix as null."},
        {"role": "user", "content": prompt}
    ],
    response_format=MathResponse
  )

  return completion.choices[0].message.content


# print(check_math_answer(user_prompt_correct))
# print(check_math_answer(user_prompt_incorrect))

def check_math_answer_no_response_format(prompt: str, client: openai.OpenAI):
    completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": """
You are are helping a student solve a math problem. Don't give away the answer, but instead guide them. They will provide you with step by step reasoning for solving a problem, and if they made a mistake, it's your job to identify the first step where they made a mistake, and provide a hint to help them correct it. Do not outright give them the answer. If they are correct, return error_step and how_to_fix as null. Format your response as a JSON object like so:
{
    "correct": "bool",
    "error_step": "int",
    "how_to_fix": "str"
}
"""},
        {"role": "user", "content": prompt}
        ],
    )
    
    json_full_string = completion.choices[0].message.content
    json_string = re.search(r"\{[^{}]*\}", json_full_string)
    return json.loads(json_string.group())


# print(check_math_answer_no_response_format(user_prompt_correct))
# print(check_math_answer_no_response_format(user_prompt_incorrect))




def check_math_answer_aimlapi(user_prompt: str, model: str, client: openai.OpenAI) -> str:

    base_url = "https://api.aimlapi.com/v1"
    api_key = api_key # free tier api-key
    system_prompt = """
You are are helping a student solve a math problem. Don't give away the answer, but instead guide them. They will provide you with step by step reasoning for solving a problem, and if they made a mistake, it's your job to identify the first step where they made a mistake, and provide a hint to help them correct it. Do not outright give them the answer. If they are correct, return error_step and how_to_fix as null. Format your response as a JSON object like so:
{
    "correct": "bool",
    "error_step": "int",
    "how_to_fix": "str"
}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    json_full_string = completion.choices[0].message.content
    json_string = re.search(r"\{[^{}]*\}", json_full_string)
    return json.loads(json_string.group())

# print(check_math_answer_no_response_format(user_prompt_correct))
# print(check_math_answer_no_response_format(user_prompt_incorrect))


