import argparse
import yaml
import random
random.seed(42)
from transformers import GenerationConfig, logging
logging.set_verbosity_error()
import os

import json
import openai
import glob

from inference import setup_model_and_tokenizer, do_initial_inference, save_results, do_correction_inference, filter_fail_initial_inference
from data_utils import load_math_dataset
from reflextion import get_corrections, show_stats, get_corrections_from_two_models
def parse_args():
    parser = argparse.ArgumentParser(description='Script configuration parameters')
    
    # Data configuration
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/MATH',
        help='Path to the data directory'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
        help='Name or path of the model to use'
    )
    
    # Hardware configuration
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run the model on (cuda or cpu)'
    )
    
    # Generation parameters
    parser.add_argument(
        '--no_sample',
        action='store_true',
        help='Disable sampling for text generation (default: sampling enabled)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        default=20,
        help='Number of samples to generate'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Temperature for text generation'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling parameter'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=2048,
        help='Maximum number of tokens to generate'
    )
    
    parser.add_argument(
        '--api_config',
        type=str,
        default='config/gpt-4o.yaml',
    )
    
    parser.add_argument(
        '--initial_inference',
        type=str,
        default='cache/initial_output.json',
    )
    
    parser.add_argument(
        '--reflexion_iters',
        type=int,
        default=3,
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    problems = load_math_dataset(args.data_path)
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.device)
    if not os.path.exists('cache'):
        os.makedirs('cache')
    
    configs = []
    api_config = args.api_config
    if api_config == "all":
        all_configs = glob.glob('config/*.yaml')
        for config in all_configs:
            with open(config, 'r') as f:
                configs.append(yaml.safe_load(f))
    else:
        with open(api_config, 'r') as f:
            api_config = yaml.safe_load(f)
        configs.append(api_config)
    
    for api_config in configs:
        sampled_problems = random.sample(problems, args.sample_size)
        api_key = api_config['api_key']
        if 'endpoint' in api_config:
            client = openai.OpenAI(api_key=api_key, base_url=api_config['endpoint'])
            client_2 = openai.OpenAI(api_key=api_config['openai_api_key'])
            has_client_2 = True
        elif api_config['model'] != "gpt-4o":
            client = openai.OpenAI(api_key=api_key)
            client_2 = openai.OpenAI(api_key=api_config['openai_api_key'])
            has_client_2 = True
        else:
            client = openai.OpenAI(api_key=api_key)
            has_client_2 = False
        
        config = GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            do_sample=not args.no_sample
        )
        if not os.path.exists(args.initial_inference):
            initial_inference_results = do_initial_inference(model, tokenizer, config, sampled_problems)
            save_results(initial_inference_results, args.initial_inference)
        with open(args.initial_inference) as f:
            initial_inference_results = json.load(f)
        
        inference_results = initial_inference_results["results"]
        inference_results, sampled_problems = filter_fail_initial_inference(inference_results, sampled_problems)
        
        for i in range(args.reflexion_iters):
            if has_client_2:
                corrections = get_corrections_from_two_models(inference_results, client, client_2, sampled_problems, model = api_config['model'])
            else:
                corrections = get_corrections(inference_results, client, sampled_problems, model = api_config['model'])
            show_stats(corrections, i)
            save_results(corrections, f'cache/corrections_{i}.json')
            corrected, problems = do_correction_inference(model, tokenizer, config, attempts=inference_results, corrections=corrections, problems=sampled_problems)
            save_results(corrected, f'cache/corrected_{i}.json')
            inference_results = corrected
            sampled_problems = problems
        
        corrections = get_corrections(inference_results, client, sampled_problems, model = api_config['model'])
        show_stats(corrections, args.reflexion_iters)

if __name__ == '__main__':
    main()