import argparse
import random
random.seed(42)
from transformers import GenerationConfig, logging
logging.set_verbosity_error()

from inference import setup_model_and_tokenizer, do_initial_inference, save_results
from data_utils import load_math_dataset

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
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    problems = load_math_dataset(args.data_path)
    sampled_problems = random.sample(problems, args.sample_size)
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.device)
    config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        do_sample=not args.no_sample
    )
    initial_inference_results = do_initial_inference(model, tokenizer, config, sampled_problems)
    save_results(initial_inference_results, 'sample_output.json')


if __name__ == '__main__':
    main()