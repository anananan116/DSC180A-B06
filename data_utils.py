import json
from pathlib import Path
from typing import Dict, List, Optional, Union

def extract_level_number(level_str: str) -> int:
    """
    Extract the numeric level from a level string.
    
    Args:
        level_str (str): Level string (e.g., "Level 3")
    
    Returns:
        int: Numeric level value
    """
    try:
        return int(level_str.lower().replace('level', '').strip())
    except (ValueError, AttributeError):
        return 0  # Return 0 for invalid or missing levels

def load_math_dataset(base_path: str, min_level: int = 1, split: str = "train") -> List[Dict]:
    """
    Load problems from the MATH dataset with a minimum difficulty level.
    
    Args:
        base_path (str): Path to the root directory containing the MATH dataset
        min_level (int): Minimum difficulty level to include (1-5)
        split (str): Dataset split to load ('train' or 'test')
    
    Returns:
        List[Dict]: List of problems at or above the minimum level, where each problem is a dictionary containing:
            - problem: The problem text
            - solution: The solution text
            - level: Difficulty level (as original string)
            - level_number: Difficulty level (as integer)
            - type: Problem category/type
    """
    if split not in ["train", "test"]:
        raise ValueError("Split must be either 'train' or 'test'")
    
    if not 1 <= min_level <= 5:
        raise ValueError("min_level must be between 1 and 5")
    
    base_path = Path(base_path)
    problems = []
    
    # MATH dataset categories
    categories = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]
    
    for category in categories:
        category_path = base_path / split / category
        if not category_path.exists():
            print(f"Warning: Path {category_path} does not exist")
            continue
            
        # Load all .json files in the category directory
        for problem_file in category_path.glob("*.json"):
            try:
                with open(problem_file, 'r', encoding='utf-8') as f:
                    problem_data = json.load(f)
                
                # Extract numeric level from level string
                level_str = problem_data.get("level", "Level 0")
                level_number = extract_level_number(level_str)
                
                # Skip problems below minimum level
                if level_number < min_level:
                    continue
                
                # Structure the problem data
                problem = {
                    "problem": problem_data["problem"],
                    "solution": problem_data["solution"],
                    "level": level_str,  # Keep original string format
                    "level_number": level_number,  # Add numeric version
                    "type": problem_data.get("type", category),
                    "file_name": problem_file.name
                }
                problems.append(problem)
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {problem_file}")
            except Exception as e:
                print(f"Error loading {problem_file}: {str(e)}")
    
    # Print summary statistics
    total_problems = len(problems)
    level_counts = {level: sum(1 for p in problems if p["level_number"] == level) 
                   for level in range(min_level, 6)}
    
    print(f"\nDataset Summary:")
    print(f"Total problems loaded: {total_problems}")
    for level, count in level_counts.items():
        print(f"Level {level} problems: {count}")
    
    return problems

def get_problem_by_id(problems: List[Dict], problem_id: str) -> Optional[Dict]:
    """
    Retrieve a specific problem by its ID (filename without extension).
    
    Args:
        problems (List[Dict]): List of problems loaded from the dataset
        problem_id (str): Problem ID to search for
    
    Returns:
        Optional[Dict]: Problem dictionary if found, None otherwise
    """
    for problem in problems:
        if problem["file_name"].replace(".json", "") == problem_id:
            return problem
    return None

def get_problems_by_level(problems: List[Dict], level: int) -> List[Dict]:
    """
    Filter problems by specific difficulty level.
    
    Args:
        problems (List[Dict]): List of problems loaded from the dataset
        level (int): Desired difficulty level (1-5)
    
    Returns:
        List[Dict]: List of problems matching the specified level
    """
    if not 1 <= level <= 5:
        raise ValueError("Level must be between 1 and 5")
    
    return [p for p in problems if p["level_number"] == level]