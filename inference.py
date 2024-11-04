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
- Start with "Solution:" on a new line
- Provide a concise answer with units if applicable
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

Solution: 3.74 years"""

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

3. FINAL ANSWER
- Start with "Solution:" on a new line
- Provide a concise answer with units if applicable
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

Solution: 3.74 years
"""

def format_chat(
    user_prompt: str, 
    system_prompt: Optional[str] = None
) -> List[Dict]:
    """Format chat messages following the standardized chat format."""
    messages = []
    
    # Add system message if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    
    # Add the current user message
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    return messages

def setup_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
):
    """Setup model and tokenizer with various quantization options."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    
    # Set up loading kwargs based on quantization settings
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    if not (load_in_8bit or load_in_4bit) and device == "cuda":
        model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def batch_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[Dict[str, str]],
    config: GenerationConfig
) -> List[Dict]:
    """Run batched inference using transformers."""
    
    # Format all prompts
    formatted_prompts = [tokenizer.apply_chat_template( 
        format_chat(
            p['user_prompt'],
            p.get('system_prompt')
        ), tokenize=False)
        for p in prompts
    ]
    
    results = []

    # Process in batches
    for i in tqdm(range(0, len(formatted_prompts), config.batch_size)):
        batch = formatted_prompts[i:i + config.batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # Adjust based on model's context window
        ).to(model.device)
        input_len = [len(i) for i in inputs['input_ids']]
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=config.do_sample
            )
        
        # Decode and store results
        for j, output in enumerate(outputs):
            original_prompt = prompts[i + j]
            generated_text = tokenizer.decode(output[input_len[j]:], skip_special_tokens=True)

            results.append({
                'user_prompt': original_prompt['user_prompt'],
                'system_prompt': original_prompt.get('system_prompt'),
                'generated_text': generated_text,
                'output_ids': output.tolist(),
                'token_count': len(output)
            })
    
    return results

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
        if line.startswith('Step') or line.startswith('step'):
            current_section = 'step'
            if one_step != '':
                steps.append(one_step)
                one_step = ''
        # Check if line starts with "Solution:"
        elif line.startswith('Solution:'):
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
    if not parsed_solution['steps']:
        return False, "No steps found in solution"
        
    # Validate step format
    step_pattern = re.compile(r'^Step \d+:', re.IGNORECASE)
    
    for step in parsed_solution['steps']:
        if not step_pattern.match(step.strip()):
            return False, f"Invalid step format: {step[:50]}..."
            
    # Validate solution format
    if not parsed_solution['solution']:
        return False, "Missing solution section"
        
    if not parsed_solution['solution'].startswith('Solution:'):
        return False, "Solution section doesn't start with 'Solution:'"
        
    return True, "Valid format"

def do_initial_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: GenerationConfig,
    dataset: List[Dict],
    max_retries: int = 10
) -> List[Dict]:
    """Run initial inference with format validation and retries."""
    # Initialize results list with None values to maintain order
    final_results = [None] * len(dataset)
    
    # Process all problems in a batch
    prompts = []
    for problem in dataset:
        prompts.append({
            'system_prompt': SYSTEM_PROMPT_INITIAL,
            'user_prompt': problem['problem']
        })
    
    # Run batch inference
    batch_results = batch_inference(model, tokenizer, prompts, config)
    
    # Process each result and handle retries if needed
    for idx, result in enumerate(batch_results):
        attempt_count = 0
        valid_result = None
        
        while attempt_count < max_retries and valid_result is None:
            if attempt_count == 0:
                # Use the initial batch result
                current_result = result
            else:
                # Retry individually for invalid results
                retry_prompt = prompts[idx]
                current_result = batch_inference(model, tokenizer, [retry_prompt], config)[0]
            
            parsed_result = parse_solution_steps(current_result['generated_text'])
            is_valid, error_msg = validate_solution_format(parsed_result)
            
            if is_valid:
                valid_result = parsed_result
            else:
                attempt_count += 1
                if attempt_count == max_retries:
                    print(f"Warning: Maximum retries reached for problem {idx}. Last error: {error_msg}")
                    valid_result = parsed_result  # Use last result even if invalid
        
        final_results[idx] = valid_result
    
    return final_results

def do_correction_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: GenerationConfig,
    attempts: List[Dict],
    corrections: List[Dict],
    problems: List[Dict],
    max_retries: int = 10
) -> Tuple[List[Dict], List[Dict]]:
    """Run correction inference with format validation and retries."""
    # Initialize results list with None for problems needing correction
    final_results = [None] * len(problems)
    final_problems = []
    correction_indices = []
    
    # Prepare prompts for problems needing correction
    prompts = []
    for idx, (attempt, correction, problem) in enumerate(zip(attempts, corrections, problems)):
        if not correction['correct']:
            user_prompt = (
                f"Problem: {problem['problem']}\n\n"
                "Previous attempt:\n"
                f"{chr(10).join(attempt['steps'][:correction['error_step']])}\n\n"
                f"Feedback: {correction['how_to_fix']}\n\n"
            )
            
            prompts.append({
                'system_prompt': SYSTEM_PROMPT_CORRECTION,
                'user_prompt': user_prompt
            })
            correction_indices.append(idx)
            final_problems.append(problem)
    
    if not prompts:  # No corrections needed
        return [], []
    
    # Run batch inference for all corrections
    batch_results = batch_inference(model, tokenizer, prompts, config)
    
    # Process each result and handle retries if needed
    for prompt_idx, (result, original_idx) in enumerate(zip(batch_results, correction_indices)):
        attempt_count = 0
        valid_result = None
        
        while attempt_count < max_retries and valid_result is None:
            if attempt_count == 0:
                # Use the initial batch result
                current_result = result
            else:
                # Retry individually for invalid results
                retry_prompt = prompts[prompt_idx]
                current_result = batch_inference(model, tokenizer, [retry_prompt], config)[0]
            
            parsed_result = parse_solution_steps(current_result['generated_text'])
            is_valid, error_msg = validate_solution_format(parsed_result)
            
            if is_valid:
                # Combine previous steps with new ones
                parsed_result['steps'] = (
                    attempts[original_idx]['steps'][:corrections[original_idx]['error_step']] + 
                    parsed_result['steps']
                )
                valid_result = parsed_result
            else:
                attempt_count += 1
                if attempt_count == max_retries:
                    print(f"Warning: Maximum retries reached for correction {prompt_idx} (original problem {original_idx}). Last error: {error_msg}")
                    # Combine steps even for invalid results
                    parsed_result['steps'] = (
                        attempts[original_idx]['steps'][:corrections[original_idx]['error_step']] + 
                        parsed_result['steps']
                    )
                    valid_result = parsed_result
        
        final_results[original_idx] = valid_result
    
    # Filter out None values from final results (problems that didn't need correction)
    final_results = [result for result in final_results if result is not None]
    
    return final_results, final_problems