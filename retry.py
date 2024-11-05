import random

# Placeholder function for the small model's response generation
def prompt_small_model(question):
    # Simulate response from a model with placeholder steps
    response = """step 1: Initial calculation based on question
step 2: Intermediate steps
step 3: Further calculation steps
Final answer: 29"""
    return response

# Step 1: Define a function to parse and verify the model's response
def parse_and_verify(response, correct_answer):
    lines = response.strip().split("\n")
    final_answer_line = lines[-1]
    try:
        final_answer = int(final_answer_line.split(":")[-1].strip())
    except ValueError:
        return False, None  # Fail gracefully if parsing fails

    # Check if the final answer matches the correct answer
    if final_answer == correct_answer:
        return True, None
    else:
        return False, lines

# Step 2: Define a function for the verification model to check each step
def verify_with_large_model(reasoning_steps, correct_answer):
    if not reasoning_steps:
        return False, None  # Avoids proceeding with empty steps

    for i, step in enumerate(reasoning_steps):
        # Placeholder logic to simulate verifying each step
        if random.choice([True, False]):  # Randomly decide a step is wrong
            error_info = {
                "error_step": i + 1,
                "reason": f"Step {i+1} reasoning error",
                "how_to_fix": f"Correction suggestion for step {i+1}"
            }
            return False, error_info
    # If all steps are correct up to the final answer
    final_answer_line = reasoning_steps[-1]
    final_answer = int(final_answer_line.split(":")[-1].strip())
    return final_answer == correct_answer, None

# Step 3: Retry with correction
def retry_with_correction(reasoning_steps, error_info):
    if not error_info or "error_step" not in error_info:
        print("Invalid error information; skipping retry.")
        return None  # Avoid proceeding without valid error information

    # Generate a revised answer more systematically (increment or decrement by 1)
    previous_answer = int(reasoning_steps[-1].split(":")[-1].strip())
    revised_answer = previous_answer + (1 if previous_answer < 4 else -1)

    # Prompt the smaller model with corrected steps and guidance
    retry_prompt = "Retry:\n" + "\n".join(reasoning_steps[:error_info["error_step"]]) + \
                   f"\n{error_info['how_to_fix']}\nFinal answer: {revised_answer}"
    print(f"Retrying with correction at step {error_info['error_step']}: {retry_prompt}")
    return retry_prompt

# Main function to handle the iterative workflow
def solve_math_question(question, correct_answer, max_iterations=3):
    iteration = 0
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}")
        # Step 1: Prompt the small model to solve
        initial_response = prompt_small_model(question)
        
        # Parse and verify
        is_correct, reasoning_steps = parse_and_verify(initial_response, correct_answer)
        
        if is_correct:
            print("Solution correct on first try.")
            return True, None
        
        # Step 2: Verify each step with the larger model
        if reasoning_steps:
            is_verified, error_info = verify_with_large_model(reasoning_steps, correct_answer)
        else:
            print("No reasoning steps provided; cannot verify.")
            return False, None

        if is_verified:
            print("Solution verified as correct.")
            return True, None
        
        # If there's an error, retry with guidance
        corrected_response = retry_with_correction(reasoning_steps, error_info)
        
        if corrected_response is None:
            print("Error in correction retry; ending process.")
            return False, None

        # Parse and verify retry
        is_correct, reasoning_steps = parse_and_verify(corrected_response, correct_answer)
        
        if is_correct:
            print("Solution corrected and verified.")
            return True, None
        
        # If incorrect, continue loop
        iteration += 1

    # If max iterations reached and still incorrect
    print("Max iterations reached. Solution not found.")
    return False, corrected_response

# Example usage:
question = "What is 2 + 2?"
correct_answer = 4

# Run the workflow
solve_math_question(question, correct_answer)
