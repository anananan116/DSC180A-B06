from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import json
from typing import List, Dict, Optional
from tqdm import tqdm
from dataclasses import dataclass
from transformers import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import re
import openai
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import concurrent.futures


logging.set_verbosity_error()

SYSTEM_PROMPT_INITIAL = """You are a precise mathematical problem solver. Follow these exact guidelines:

1. READ AND UNDERSTAND
- Begin by restating the problem in your own words
- List all given values and what you need to find
- If relevant, mention any formulas you'll need to use

2. SOLUTION FORMAT
- Present each logical step of your solution sequentially
- Each step must start with "Step n:" on a new line, where n is the step number
- After "Step n:", provide a clear explanation of what you're doing and why
- Show all mathematical operations
- Each step should be relatively short and focused
- Even if you find an error in previous steps, continue to the end. Do not fix it and start again!

3. FINAL ANSWER
- Start with "\\boxed{}" on a new line, putting the final answer in the brackets
- Do not include any additional information in the final answer, inlcudding units
- Still provide the answer even if you made a mistake in the steps, even if it's incorrect or not valid. NEVER start again!

Example output:
Problem: How long will it take for $1000 to grow to $1200 at 5% annual interest?

Given:
- Principal (P) = $1000
- Future Value (A) = $1200
- Interest Rate (r) = 5% = 0.05
- Formula: A = P(1 + r)^t

Step 1: Set up the equation
1200 = 1000(1 + 0.05)^t

Step 2: Divide both sides by 1000
1.2 = (1.05)^t

Step 3: Take natural log of both sides
ln(1.2) = t x ln(1.05)

Step 4: Solve for t
t = ln(1.2) ÷ ln(1.05)
t = 3.74

\\boxed{3.74}"""

SYSTEM_PROMPT_CORRECTION = """You are a precise mathematical problem solver. Follow these exact guidelines:

1. READ AND UNDERSTAND
- Begin by acknowledging the previous attempt provided
- The last step shown in the previous attempt is always the incorrect step
- State which step you're starting from (the last step shown) and explain why it needs correction based on the feedback
- List any additional information from the feedback that will help fix the solution

2. SOLUTION FORMAT FOR CORRECTIONS
- Start from the step number of the last step shown in the previous attempt
- Begin with "Step n:" where n is the last step number shown
- Show the corrected mathematical operations
- Continue with subsequent steps as normal, incrementing the step number
- Each step must start with "Step n:" on a new line
- Provide clear explanation of what you're doing and why
- Show all mathematical operations
- Each step should be relatively short and focused
- Even if you find an error in previous steps, continue to the end. Do not fix it and start again!
- No matter what the feedback and the previous attempt is, you should always cotinue solving the problem following the given format and provide a final answer!
- Do not repeat the original wrong step in the corrected solution. Simply start from the corrected step.

3. FINAL ANSWER
- Start with "\\boxed{}" on a new line, putting the final answer in the brackets
- Do not include any additional information in the final answer, inlcudding units
- Still provide the answer even if you made a mistake in the steps, even if it's incorrect or not valid. NEVER start again!
- No matter what the feedback and the previous attempt is, you should always cotinue solving the problem following the given format and provide a final answer!

Example input:
Problem: How long will it take for $1000 to grow to $1200 at 5% annual interest?

Previous attempt:
Step 1: Set up the equation
1200 = 1000(1 + 0.05)^t

Step 2: Divide both sides by 1000
1.2 = (1.05)^t

Step 3: Take natural log of both sides
ln(1.2) = t + ln(1.05)

Feedback: After taking log, the t should be multiplied by ln(1.05), not added.

Example output:

Step 3: Take natural log of both sides
ln(1.2) = t x ln(1.05)

Step 4: Solve for t
t = ln(1.2) ÷ ln(1.05)
t = 3.74

\\boxed{3.74}
"""

def save_results(results: List[Dict], output_file: str):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump({'results': results}, f)

def parse_solution_steps(solution_text) -> Dict:
    """
    Parse a solution text into individual steps.
    
    Args:
        solution_text (str): The solution text containing the problem and steps
        
    Returns:
        steps: A list of strings, each representing a step
        solutions: A string representing the solution
    """
    # Split the text by newlines and clean up
    lines = [line.strip() for line in solution_text.split('\n') if line.strip()]
    
    # Initialize lists for different parts
    steps = []
    solution = None
    current_section = 'none'
    one_step = ''
    for line in lines:
        if line.replace('*', '').startswith('Step') or line.replace('*', '').startswith('step'):
            current_section = 'step'
            if one_step != '':
                steps.append(one_step)
                one_step = ''
        # Check if line starts with "\boxed{}:"
        elif '\\boxed{' in line:
            solution = line
            break
        if current_section == 'step':
            one_step += line + '\n'
    return {
        'steps': steps,
        'solution': solution
    }

def validate_solution_format(parsed_solution: Dict) -> Tuple[bool, str]:
    """
    Validate if the solution follows the expected format.
    
    Args:
        parsed_solution (Dict): Dictionary containing 'steps' and 'solution'
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if 'steps' not in parsed_solution or 'solution' not in parsed_solution:
        return False, "Missing required sections: steps or solution"
        
    # Check if there are any steps
    if len(parsed_solution['steps']) == 0 or len(parsed_solution['steps']) == 1:
        return False, "No steps found in solution"
            
    # Validate solution format
    if not parsed_solution['solution']:
        return False, "Missing solution section"
        
    return True, "Valid format"

def process_single_inference(args: tuple) -> Dict:
    """Process a single inference request using the OpenAI API."""
    problem, model, client, system_prompt, original_index = args
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem['user_prompt']},
            ],
            max_tokens=4096
        )
        
        return {
            'generated_text': completion.choices[0].message.content,
            'original_index': original_index
        }
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return {'generated_text': None, 'original_index': original_index}

def batch_api_inference(
    client: openai.OpenAI,
    model: str,
    prompts: List[Dict],
    system_prompt: str,
    max_workers: int = 128
) -> List[Dict]:
    """Run batch inference using concurrent API calls."""
    # Include original index in task arguments
    task_args = [(prompt, model, client, system_prompt, idx) for idx, prompt in enumerate(prompts)]
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_inference, args) for args in task_args]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(task_args)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Include the original index even for failed results
                original_idx = task_args[futures.index(future)][-1]
                results.append({'generated_text': None, 'original_index': original_idx})
    
    # Sort results based on original index
    sorted_results = sorted(results, key=lambda x: x['original_index'])
    # Remove the original_index from final results
    final_results = [{'generated_text': r['generated_text']} for r in sorted_results]
    
    return final_results

def do_initial_inference(
    client: openai.OpenAI,
    model: str,
    dataset: List[Dict],
    max_retries: int = 3
) -> List[Dict]:
    """Run initial inference with retries using the API."""
    final_results = [None] * len(dataset)
    
    # Prepare prompts
    prompts = []
    for problem in dataset:
        prompts.append({
            'user_prompt': problem['problem']
        })
    
    # Run batch inference
    batch_results = batch_api_inference(client, model, prompts, SYSTEM_PROMPT_INITIAL)
    
    # Process results and handle retries
    for idx, result in enumerate(batch_results):
        attempt_count = 0
        valid_result = None
        
        while attempt_count < max_retries and valid_result is None:
            if attempt_count == 0:
                current_result = result
            else:
                print(f"Retrying initial generation: problem {idx}")
                # For retries, we need to maintain the original index
                retry_prompt = prompts[idx]
                current_result = process_single_inference((retry_prompt, model, client, SYSTEM_PROMPT_INITIAL, 0))
            
            if current_result['generated_text'] is not None:
                parsed_result = parse_solution_steps(current_result['generated_text'])
                valid_result = parsed_result
            else:
                attempt_count += 1
                if attempt_count == max_retries:
                    print(f"Warning: Maximum retries reached for problem {idx}")
                    valid_result = None
        
        final_results[idx] = valid_result
    
    return final_results

def do_correction_inference(
    client: openai.OpenAI,
    model: str,
    attempts: List[Dict],
    corrections: List[Dict],
    problems: List[Dict],
    index_map: List[int],
    max_retries: int = 3
) -> Tuple[List[Dict], List[Dict], List[int]]:
    """Run correction inference with retries using the API."""
    # Initialize data structures to maintain order
    correction_data = []
    final_problems = []
    filtered_index = []
    
    # Prepare prompts for corrections and track indices
    for idx, (attempt, correction, problem) in enumerate(zip(attempts, corrections, problems)):
        if not correction['correct']:
            user_prompt = (
                f"Problem: {problem['problem']}\n\n"
                "Previous attempt:\n"
                f"{chr(10).join(attempt['steps'][:correction['error_step']])}\n\n"
                f"Feedback: {correction['how_to_fix']}\n\n"
            )
            
            correction_data.append({
                'prompt': {'user_prompt': user_prompt},
                'original_idx': idx,
                'attempt': attempt,
                'correction': correction,
                'problem': problem,
                'map_idx': index_map[idx]
            })
    
    if not correction_data:
        return [], [], []
    
    # Extract prompts while maintaining order information
    prompts = [item['prompt'] for item in correction_data]
    
    # Run batch inference with tracking indices
    batch_results = batch_api_inference(
        client, 
        model, 
        prompts, 
        SYSTEM_PROMPT_CORRECTION
    )
    
    final_results = []
    
    # Process results and handle retries while maintaining order
    for idx, (result, correction_item) in enumerate(zip(batch_results, correction_data)):
        attempt_count = 0
        valid_result = None
        
        while attempt_count < max_retries and valid_result is None:
            if attempt_count == 0:
                current_result = result
            else:
                print(f"Retrying correction: {idx} (original problem {correction_item['original_idx']})")
                current_result = process_single_inference((correction_item['prompt'], model, client, SYSTEM_PROMPT_CORRECTION, 0))
            
            if current_result['generated_text'] is not None:
                parsed_result = parse_solution_steps(current_result['generated_text'])
                # Combine previous steps with new ones
                parsed_result['steps'] = (
                    correction_item['attempt']['steps'][:correction_item['correction']['error_step'] - 1] + 
                    parsed_result['steps']
                )
                valid_result = parsed_result
            else:
                attempt_count += 1
                if attempt_count == max_retries:
                    print(f"Warning: Maximum retries reached for correction {idx} (original problem {correction_item['original_idx']})")
                    # Combine steps even for invalid results
                    parsed_result = {
                        'steps': correction_item['attempt']['steps'][:correction_item['correction']['error_step']]
                    }
                    valid_result = parsed_result
        
        if valid_result is not None:
            final_results.append(valid_result)
            final_problems.append(correction_item['problem'])
            filtered_index.append(correction_item['map_idx'])
    
    return final_results, final_problems, filtered_index

def filter_fail_initial_inference(results: List[Dict], problems: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Filter out failed initial inference results."""
    valid_results = []
    valid_problems = []
    for result, problem in zip(results, problems):
        if result is not None:
            valid_results.append(result)
            valid_problems.append(problem)
    print(f"Filtered out {len(results) - len(valid_results)} failed initial inference results.")
    return valid_results, valid_problems