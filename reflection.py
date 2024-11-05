import openai
import os
from typing import Optional, TypedDict
import pydantic
import json
import re
from tqdm import tqdm
import concurrent.futures

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

Please first provide a step-by-step check of the student's solution. When you identify an error or confirm the solution is correct, provide the feedback. When providing feedback, please consider the following:
- Clearly identify the step number where the error occurred
- Provide a clear and concise explanation of the error
- Provide a hint to help the student correct the error
- DO NOT provide the correct solution to the problem, only hints to help the student correct their mistake
- Follow the following format in to provide the feedback in the end of your response:

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

def process_single_correction(args: tuple) -> CorrectionResponse:
    result, problem, model, client = args
    problem_text = problem["problem"]
    solution = problem["solution"]
    problem_prompt = f"Problem: {problem_text}\n\nCorrect Solution: {solution}\n\nStudent's Solution:\n"
    
    if result["solution"] is None:
        result["solution"] = "null"
    
    student_solution = "\n".join(result["steps"]) + "\n" + result["solution"]
    user_prompt = problem_prompt + student_solution
    return check_math_answer(user_prompt, model, client)

def get_corrections(results: list[dict], client: openai.OpenAI, problems: list[dict], model: str) -> list[dict]:
    # Create arguments for each task
    task_args = [(result, problem, model, client) for result, problem in zip(results, problems)]
    
    corrections = []
    # Use ThreadPoolExecutor for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        # Create future objects for all tasks
        future_to_task = {executor.submit(process_single_correction, args): args for args in task_args}
        
        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(task_args)):
            try:
                correction = future.result()
                corrections.append(correction)
            except Exception as e:
                # Handle any errors and append a default error response
                corrections.append({
                    "correct": False,
                    "error_step": 1,
                    "how_to_fix": f"Error processing correction: {str(e)}"
                })
    
    # Sort corrections back into original order
    sorted_corrections = [None] * len(results)
    for idx, (future, args) in enumerate(future_to_task.items()):
        original_idx = task_args.index(args)
        sorted_corrections[original_idx] = corrections[idx]
    
    return sorted_corrections

def get_corrections_from_two_models(
    results: list[dict],
    client: openai.OpenAI,
    client_2: openai.OpenAI,
    problems: list[dict],
    model: str
) -> tuple[list[dict]]:
    # Create arguments for each task for both models
    task_args_1 = [(result, problem, model, client) for result, problem in zip(results, problems)]
    task_args_2 = [(result, problem, "gpt-4o", client_2) for result, problem in zip(results, problems)]
    
    corrections_1 = []
    corrections_2 = []
    
    # Use ThreadPoolExecutor for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks for both models
        future_to_task_1 = {executor.submit(process_single_correction, args): args for args in task_args_1}
        future_to_task_2 = {executor.submit(process_single_correction, args): args for args in task_args_2}
        
        # Combine all futures for progress tracking
        all_futures = list(future_to_task_1.keys()) + list(future_to_task_2.keys())
        
        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(all_futures), total=len(all_futures)):
            try:
                correction = future.result()
                if future in future_to_task_1:
                    corrections_1.append(correction)
                else:
                    corrections_2.append(correction)
            except Exception as e:
                error_response = {
                    "correct": False,
                    "error_step": 1,
                    "how_to_fix": f"Error processing correction: {str(e)}"
                }
                if future in future_to_task_1:
                    corrections_1.append(error_response)
                else:
                    corrections_2.append(error_response)
    
    # Sort corrections back into original order
    sorted_corrections_1 = [None] * len(results)
    sorted_corrections_2 = [None] * len(results)
    
    for idx, (future, args) in enumerate(future_to_task_1.items()):
        original_idx = task_args_1.index(args)
        sorted_corrections_1[original_idx] = corrections_1[idx]
        
    for idx, (future, args) in enumerate(future_to_task_2.items()):
        original_idx = task_args_2.index(args)
        sorted_corrections_2[original_idx] = corrections_2[idx]
    
    return sorted_corrections_1, sorted_corrections_2

def validate_json_response(json_data: dict) -> CorrectionResponse:
    try:
        validated_data = ResponseValidator(**json_data)
        return validated_data.dict()
    except pydantic.ValidationError as e:
        raise ValueError(f"Invalid JSON structure: {str(e)}")

def find_json_in_text(text: str) -> Optional[str]:
    """
    Find JSON object in text containing LaTeX equations by using a state machine approach.
    This handles nested braces and ignores LaTeX equation blocks.
    """
    in_latex = False
    brace_count = 0
    start_pos = -1
    escape_next = False
    
    for i, char in enumerate(text):
        # Handle LaTeX equation markers
        if char == '$' and not escape_next:
            in_latex = not in_latex
            continue
            
        # Skip processing if we're inside a LaTeX equation
        if in_latex:
            continue
            
        # Handle escape characters
        if char == '\\':
            escape_next = not escape_next
            continue
        escape_next = False
        
        # Track JSON structure
        if char == '{' and not in_latex:
            if brace_count == 0:
                start_pos = i
            brace_count += 1
        elif char == '}' and not in_latex:
            brace_count -= 1
            if brace_count == 0 and start_pos != -1:
                # Found a potential complete JSON object
                candidate = text[start_pos:i+1]
                try:
                    # Verify it's valid JSON
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    # Reset and continue searching
                    start_pos = -1
                    
    return None

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
            json_string = find_json_in_text(json_full_string)
            
            if not json_string:
                raise ValueError("No JSON object found in response")
            
            json_loaded = json.loads(json_string)
            validated_response = validate_json_response(json_loaded)
            return validated_response
            
        except (json.decoder.JSONDecodeError, ValueError, pydantic.ValidationError) as e:
            retry_count += 1
            if retry_count == max_retries:
                return CorrectionResponse(
                    correct=False,
                    error_step=1,
                    how_to_fix="Unable to parse response from model"
                )

def show_stats(corrections: list[dict], iter: int = 0, model: str = "gpt-4o"):
    total_problems = len(corrections)
    correct_count = sum(1 for c in corrections if c["correct"])
    incorrect_count = total_problems - correct_count

    report = f"Round {iter} Summary for {model}:\n"
    report += f"Total problems: {total_problems}\n"
    report += f"Correct solutions: {correct_count}\n"
    report += f"Incorrect solutions: {incorrect_count}\n"
    print(report)
    model = model.replace("/", "_")
    if not os.path.exists(f"reports/{model}_summary.txt"):
        with open(f"reports/{model}_summary.txt", "w") as f:
            f.write(report)
    else:
        with open(f"reports/{model}_summary.txt", "a") as f:
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
    mismatch_passthrough = sum(1 for c1, c2 in zip(correctness_1, correctness_2) if (c1 and (not c2)))
    
    report = f"Round {iter} Summary for {model}:\n"
    report += f"Total problems: {total_problems}\n"
    report += f"Correct solutions (Model 1): {correct_count}\n"
    report += f"Incorrect solutions (Model 1): {incorrect_count}\n"
    report += f"Correct solutions (Model 2): {correct_count_2}\n"
    report += f"Mismatched solutions: {mismatch_count}\n"
    report += f"Mismatched solutions leaved behind: {mismatch_passthrough}\n"
    print(report)
    model = model.replace("/", "_")
    if not os.path.exists(f"reports/{model}_summary.txt"):
        with open(f"reports/{model}_summary.txt", "w") as f:
            f.write(report)
    else:
        with open(f"reports/{model}_summary.txt", "a") as f:
            f.write(report)
    return report