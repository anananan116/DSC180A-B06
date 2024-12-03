import openai
import os
from typing import Optional, TypedDict
import pydantic
import json
import re
from tqdm import tqdm
import concurrent.futures
from eval import EvaluatorMathBatch, RespSampleBase
SYSTEM_PROMPT = """
You will be given a math problem, the correct solution and the solution provided by a student. 
You should check the student's solution step-by-step to identify any errors.
Your task is to identify the first step where the student made a mistake and provide a hint to help them correct it. 
If the student's solution is correct, return error_step and how_to_fix as null.

When determining whether the student's solution is correct according to the given ground truth answer. The answer by the student doesn't need to match the ground truth exactly, but it should be logically equivalent.

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

def remove_latex(text: str) -> str:
    if text is None:
        return ""
    if "\\boxed{" in text:
        text = text.split("\\boxed{")[1]
        if text[-1] == "}":
            text.replace("}", "")
    return text

def process_single_correction(args: tuple) -> CorrectionResponse:
    result, problem, model, client = args
    problem_text = problem["problem"]
    solution = problem["answer"]
    problem_prompt = f"Problem: {problem_text}\n\nStudent's Solution:\n"
    
    if result["solution"] is None:
        result["solution"] = "null"
    
    student_solution = "\n".join(result["steps"]) + "\n" + "Solution: "+ remove_latex(result["solution"])
    user_prompt = problem_prompt + student_solution + f"\n\nCorrect Answer: {solution}"
    return check_math_answer(user_prompt, model, client)

def get_corrections(results: list[dict], client: openai.OpenAI, problems: list[dict], model: str) -> list[dict]:
    # Create arguments for each task
    task_args = [(result, problem, model, client) for result, problem in zip(results, problems)]
    
    corrections = []
    # Use ThreadPoolExecutor for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
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
import pickle

def show_stats(corrections: list[dict], currect_solutions: dict, original_problems: list[dict], iter: int = 0, model: str = "gpt-4o"):
    gt_solutions = [x["answer"] for x in original_problems]
    new_currect_solutions = []
    for i in range(len(gt_solutions)):
        if "solution" in currect_solutions[i]:
            new_currect_solutions.append(currect_solutions[i]["solution"])
        else:
            new_currect_solutions.append("null")
    samples = []
    for currect_soltion, gt_solution in zip(new_currect_solutions, gt_solutions):
        samples.append(RespSampleBase(ref_ans=gt_solution, resp=currect_soltion))
    evaluator = EvaluatorMathBatch()
    answers, corrects = evaluator.batch_eval(samples)
    
    total_correct_dart  = sum(corrects)
    
    total_problems = len(corrections)
    correct_count = sum(1 for c in corrections if c["correct"])
    incorrect_count = total_problems - correct_count

    report = f"Round {iter} Summary for {model}:\n"
    report += f"Total problems: {total_problems}\n"
    report += f"Correct solutions: {correct_count}\n"
    report += f"Incorrect solutions: {incorrect_count}\n"
    report += f"=============================\n"
    report += f"Total correct solutions by Dart: {total_correct_dart}\n"
    print(report)
    model = model.replace("/", "_")
    if not os.path.exists(f"reports/{model}_summary.txt"):
        with open(f"reports/{model}_summary.txt", "w") as f:
            f.write(report)
    else:
        with open(f"reports/{model}_summary.txt", "a") as f:
            f.write(report)
    return report