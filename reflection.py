import openai
import os
from typing import Optional, TypedDict
import pydantic
import json
import re
from tqdm import tqdm

SYSTEM_PROMPT = """
You will be given a math problem, the correct solution and the solution provided by a student. 
You should check the student's solution step-by-step to identify any errors.
Your task is to identify the first step where the student made a mistake and provide a hint to help them correct it. 
If the student's solution is correct, return error_step and how_to_fix as null.

When determining whether the student's solution is correct, consider the following:
- Each step should be logically connected to the previous step
- Each step should be mathematically sound
- Each step should be written clearly and legibly
- The overall solution should be correct

Please first provide a step-by-step check of the student's solution. When you identify an error or confirm the solution is correct, provide the feedback in a new line with the following format:

{
    "correct": "bool",
    "error_step": "int",
    "how_to_fix": "str"
}
"""

class CorrectionResponse(TypedDict):
    correct: bool
    error_step: Optional[int]
    how_to_fix: Optional[str]

class ResponseValidator(pydantic.BaseModel):
    correct: bool
    error_step: Optional[int] = None
    how_to_fix: Optional[str] = None

    @pydantic.validator('error_step')
    def validate_error_step(cls, v, values):
        if not values.get('correct', True) and v is None:
            raise ValueError("error_step must be provided when correct is False")
        if values.get('correct', False) and v is not None:
            raise ValueError("error_step must be None when correct is True")
        return v

    @pydantic.validator('how_to_fix')
    def validate_how_to_fix(cls, v, values):
        if not values.get('correct', True) and v is None:
            raise ValueError("how_to_fix must be provided when correct is False")
        if values.get('correct', False) and v is not None:
            raise ValueError("how_to_fix must be None when correct is True")
        return v

def get_corrections(results: list[dict], client: openai.OpenAI, problems: list[dict], model: str) -> list[dict]:
    corrections = []
    for result, problem in tqdm(zip(results, problems), total=len(results)):
        problem_text = problem["problem"]
        solution = problem["solution"]
        problem_prompt = f"Problem: {problem_text}\n\nCorrect Solution: {solution}\n\nStudent's Solution:\n"
        if result["solution"] is None:
            result["solution"] = "null"
        student_solution = "\n".join(result["steps"]) + "\n" + result["solution"]
        user_prompt = problem_prompt + student_solution
        correction_json = check_math_answer(user_prompt, model, client)
        corrections.append(correction_json)
    return corrections

def get_corrections_from_two_models(results: list[dict], client: openai.OpenAI, client_2:openai.OpenAI, problems: list[dict], model: str) -> tuple[list[dict]]:
    corrections = []
    corrections_2 = []
    for result, problem in tqdm(zip(results, problems), total=len(results)):
        problem_text = problem["problem"]
        solution = problem["solution"]
        problem_prompt = f"Problem: {problem_text}\n\nCorrect Solution: {solution}\n\nStudent's Solution:\n"
        if result["solution"] is None:
            result["solution"] = "null"
        student_solution = "\n".join(result["steps"]) + "\n" + result["solution"]
        user_prompt = problem_prompt + student_solution
        correction_json = check_math_answer(user_prompt, model, client)
        correction_json_2 = check_math_answer(user_prompt, "gpt-4o", client_2)
        corrections_2.append(correction_json_2)
        corrections.append(correction_json)
    return corrections, corrections_2

def validate_json_response(json_data: dict) -> CorrectionResponse:
    try:
        # Validate using pydantic model
        validated_data = ResponseValidator(**json_data)
        return validated_data.dict()
    except pydantic.ValidationError as e:
        raise ValueError(f"Invalid JSON structure: {str(e)}")

def check_math_answer(user_prompt: str, model: str, client: openai.OpenAI) -> CorrectionResponse:
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            json_full_string = completion.choices[0].message.content
            json_string = re.search(r"\{[^{}]*\}", json_full_string)
            if not json_string:
                raise ValueError("No JSON object found in response")
            
            json_loaded = json.loads(json_string.group())
            # Validate JSON structure and types
            validated_response = validate_json_response(json_loaded)
            return validated_response
            
        except (json.decoder.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            retry_count += 1
            if retry_count == max_retries:
                # Return a default error response if all retries fail
                return {
                    "correct": False,
                    "error_step": 1,
                    "how_to_fix": "Unable to parse response from model"
                }

def show_stats(corrections: list[dict], iter: int = 0, model: str = "gpt-4o"):
    total_problems = len(corrections)
    correct_count = sum(1 for c in corrections if c["correct"])
    incorrect_count = total_problems - correct_count

    report = f"Round {iter} Summary for {model}:\n"
    report += f"Total problems: {total_problems}\n"
    report += f"Correct solutions: {correct_count}\n"
    report += f"Incorrect solutions: {incorrect_count}\n"
    print(report)
    with open(f"{model}_summary.txt", "a") as f:
        f.write(report)
    return report

def show_stats_two_models(corrections: list[dict], corrections_2: list[dict], iter: int = 0, model: str = "gpt-4o"):
    correctness_1 = [c["correct"] for c in corrections]
    correctness_2 = [c["correct"] for c in corrections_2]
    total_problems = len(corrections)
    correct_count = sum(1 for c in correctness_1 if c)
    correct_count_2 = sum(1 for c in correctness_2 if c)
    incorrect_count = total_problems - correct_count
    mismatch_count = sum(1 for c1, c2 in zip(correctness_1, correctness_2) if (c1 != c2))
    
    report = f"Round {iter} Summary for {model}:\n"
    report += f"Total problems: {total_problems}\n"
    report += f"Correct solutions (Model 1): {correct_count}\n"
    report += f"Incorrect solutions (Model 1): {incorrect_count}\n"
    report += f"Correct solutions (Model 2): {correct_count_2}\n"
    report += f"Mismatched solutions: {mismatch_count}\n"
    print(report)
    with open(f"{model}_summary.txt", "a") as f:
        f.write(report)
    return report