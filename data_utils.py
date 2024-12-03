import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import load_dataset

def load_math_dataset(base_path: str, min_level: int = 1, split: str = "train") -> List[Dict]:
    """
    Load MATH 500 subset of the HuggingFaceH4 dataset.
    """
    ds = load_dataset("HuggingFaceH4/MATH-500")
    data = ds['test']
    data_list = []
    for i in range(len(data)):
        problem_data = data[i]
        problem = {"problem": problem_data["problem"],
                    "solution": problem_data["solution"],
                    "answer": problem_data["answer"],
                    "level": problem_data["level"],
                    "type": problem_data["subject"],
                    "file_name": problem_data["unique_id"]}
        data_list.append(problem)
    
    return data_list
