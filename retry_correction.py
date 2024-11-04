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
