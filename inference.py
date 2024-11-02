from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import json
from typing import List, Dict, Optional
from tqdm import tqdm
from dataclasses import dataclass
from transformers import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logging.set_verbosity_error()

SYSTEM_PROMPT = """You are a precise mathematical problem solver. Follow these exact guidelines:

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
ln(1.2) = t × ln(1.05)

Step 4: Solve for t
t = ln(1.2) ÷ ln(1.05)
t = 3.74

Solution: 3.74 years"""

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

def do_initial_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: GenerationConfig,
    dataset: List[Dict]
) -> List[Dict]:
    """Run initial inference and return results."""
    prompts = []
    for one_problem in dataset:
        one_peompt = {}
        one_peompt['system_prompt'] = SYSTEM_PROMPT
        one_peompt['user_prompt'] = one_problem['problem']
        prompts.append(one_peompt)
    results = batch_inference(model, tokenizer, prompts, config)
    generated = [x['generated_text'] for x in results]
    results = [parse_solution_steps(x) for x in generated]
    return results