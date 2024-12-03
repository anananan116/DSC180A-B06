import argparse
import yaml
import random
random.seed(42)
from transformers import GenerationConfig, logging
logging.set_verbosity_error()
import os
import copy
import json
import openai
import glob

from inference import do_initial_inference, save_results, do_correction_inference, filter_fail_initial_inference
from data_utils import load_math_dataset
from reflection import get_corrections, show_stats

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
        default=500,
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
        default=['config/llama-70b.yaml', 'config/llama-405b.yaml'],
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
    if not os.path.exists('cache'):
        os.makedirs('cache')
    if not os.path.exists('reports'):
        os.makedirs('reports')
    initial_sampled_dataset = problems[:args.sample_size]
    configs = []
    api_config = args.api_config
    for config in api_config:
        with open(config, 'r') as f:
            configs.append(yaml.safe_load(f))
    # if api_config == "all":
    #     all_configs = glob.glob('config/*.yaml')
    #     for config in all_configs:
    #         with open(config, 'r') as f:
    #             configs.append(yaml.safe_load(f))
    # else:
    #     with open(api_config, 'r') as f:
    #         api_config = yaml.safe_load(f)
    #     configs.append(api_config)
    
    for api_config in configs:
        sampled_problems = initial_sampled_dataset
        api_key = api_config['api_key']
        if 'endpoint' in api_config:
            client = openai.OpenAI(api_key=api_key, base_url=api_config['endpoint'])
        else:
            client = openai.OpenAI(api_key=api_key)
        
        if not os.path.exists(args.initial_inference):
            initial_inference_results = do_initial_inference(client = client, model = "meta-llama/Meta-Llama-3-8B-Instruct-Turbo", dataset= sampled_problems)
            save_results(initial_inference_results, args.initial_inference)
        with open(args.initial_inference) as f:
            initial_inference_results = json.load(f)
        
        inference_results = initial_inference_results["results"]
        inference_results, sampled_problems = filter_fail_initial_inference(inference_results, sampled_problems)
        
        original_problems = copy.deepcopy(sampled_problems)
        save_results(original_problems, "cache/original_problems.json")
        currect_solutions = {i:v for i, v in enumerate(inference_results)}
        index_map = range(len(inference_results))
        
        for i in range(args.reflexion_iters):
            corrections = get_corrections(inference_results, client, sampled_problems, model = api_config['model'])
            report = show_stats(corrections, currect_solutions, original_problems, i, api_config['model'])
            save_results(corrections, f'cache/corrections_{i}.json')
            corrected, problems, index_map = do_correction_inference(client=client, model= api_config['model'], attempts=inference_results, corrections=corrections, problems=sampled_problems, index_map=index_map)
            for i, v in enumerate(corrected):
                currect_solutions[index_map[i]] = v
            save_results(corrected, f'cache/corrected_{i}.json')
            inference_results = corrected
            sampled_problems = problems
        
        corrections = get_corrections(inference_results, client, sampled_problems, model = api_config['model'])
        show_stats(corrections, currect_solutions, original_problems, i, api_config['model'])

if __name__ == '__main__':
    main()