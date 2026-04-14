"""
Per-ingredient portion estimation for Nutrition5k dataset using FNDDS standard units.
Uses parallel processing by default for faster execution.

Input: Output from nutrition5k_food_code.py (nutrition5k_results.csv)
Output: GPTAmount column with per-ingredient portions in format:
    "ingredient1: 1.5 cup
    ingredient2: 2 tablespoon
    ..."
And GPTFoodCode formatted as:
    "ingredient1: code1; ingredient2: code2; ..."

Compatible with nutrient_estimate_mix.py for final nutrient calculation.
"""
import pandas as pd
import numpy as np
import os
import time
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from chagApp_openai import Vision


def image_to_base64_url(image_path):
    """Convert a local image file to a base64 data URL."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.lower().endswith('.png'):
        mime_type = 'image/png'
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = 'image/jpeg'
    else:
        mime_type = 'image/png'

    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')

    return f"data:{mime_type};base64,{encoded}"


def load_checkpoint(checkpoint_path):
    """Load the last processed index from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as file:
            content = file.read().strip()
            if content:
                return int(content)
    return 0


def save_checkpoint(index, checkpoint_path):
    """Save the current index to a checkpoint file."""
    with open(checkpoint_path, 'w') as file:
        file.write(str(index))
    print(f"Checkpoint saved: processed up to index {index}")


def parse_food_codes(food_code_str):
    """
    Parse GPTFoodCode from new format: "ingredient1: code1; ingredient2: code2; ..."
    Example: "chicken: 27446400; rice: 20081000" -> ['27446400', '20081000']
    """
    if pd.isna(food_code_str) or food_code_str == '':
        return []

    try:
        # Split by semicolon to get individual "ingredient: code" pairs
        pairs = food_code_str.split(';')
        codes = []

        for pair in pairs:
            if ':' in pair:
                # Extract code after the colon
                parts = pair.split(':', 1)
                code = parts[1].strip()
                # Filter out "unknown"
                if code and code != "unknown":
                    codes.append(code)

        return codes
    except Exception as e:
        print(f"Error parsing food codes: {food_code_str}, Error: {e}")
        return []


def parse_ingredients_from_gpt_description(gpt_description):
    """
    Parse ingredients from GPTFoodDescription (multi-line format from nutrition5k_food_code.py)
    Example:
    "grilled chicken
    white rice
    steamed broccoli" -> ['grilled chicken', 'white rice', 'steamed broccoli']
    """
    if pd.isna(gpt_description) or gpt_description == '':
        return []

    # Split by newline and strip whitespace
    ingredients = [ing.strip() for ing in str(gpt_description).split('\n') if ing.strip()]

    # Filter out empty lines and error messages
    ingredients = [ing for ing in ingredients if ing and not ing.startswith("I can't")]

    return ingredients


def create_per_ingredient_portion_prompt(food_description, ingredient, portion_options):
    """
    Create prompt for estimating portion size for a single ingredient using FNDDS standard units.
    Similar to rag_portion_size.py but for individual ingredients.
    """
    return f"""The image displays food described as: {food_description}

Please estimate the portion size for this specific ingredient: {ingredient}

Available portion sizes for this food:
{portion_options}

Please respond with:
1. First line: The amount (a number, can be decimal like 1.5, 0.5, 2, etc.)
2. Second line: The unit (exactly as shown in the options above)

Example response:
1.5
cup

If you cannot estimate, reply with "unknown" on both lines.
If you cannot analyze the image, reply with "I can't help to analyze this text." and specify the reason on a new line."""


def match_ingredients_to_codes(ingredients_list, food_codes_list):
    """
    Create formatted string for GPTFoodCode column compatible with nutrient_estimate_mix.py

    Args:
        ingredients_list: ['chicken', 'rice', 'broccoli']
        food_codes_list: ['27446400', '20081000', '11090000']

    Returns:
        Formatted string: "chicken: 27446400; rice: 20081000; broccoli: 11090000"
    """
    if len(ingredients_list) == 1 and len(food_codes_list) == 1:
        # Single ingredient - simple format
        return f"{ingredients_list[0]}: {food_codes_list[0]}"

    # Multiple ingredients - distribute codes
    matched = []
    for i, ingredient in enumerate(ingredients_list):
        if i < len(food_codes_list):
            code = food_codes_list[i]
        else:
            # Use last code for remaining ingredients
            code = food_codes_list[-1] if food_codes_list else "00000000"
        matched.append(f"{ingredient}: {code}")

    return "; ".join(matched)



def process_single_dish_portions(args):
    """Process a single dish for portion estimation - used for parallel processing."""
    index, dish_id, image_path, food_description, ingredients_list, food_codes_list, portion_weights_df, model_name = args

    try:
        # Create thread-local Vision instance
        llm = Vision(model_name)

        # Convert image to base64
        image_data = image_to_base64_url(image_path)
        portion_results = []

        # Estimate portion for each ingredient
        for idx, ingredient in enumerate(ingredients_list):
            code = food_codes_list[idx] if idx < len(food_codes_list) else None

            if not code or code == "unknown":
                portion_results.append(f"{ingredient}: unknown")
                continue

            # Get available portions for this food code
            try:
                food_code_int = int(code)
                portions = portion_weights_df[portion_weights_df['FoodCode'] == food_code_int]['Portion'].dropna().unique()

                if len(portions) == 0:
                    portion_results.append(f"{ingredient}: unknown")
                    continue

                # Format portion options
                portion_options = ', '.join([str(p) for p in portions[:10]])  # Limit to 10 options

                # Ask GPT for this specific ingredient
                prompt = create_per_ingredient_portion_prompt(food_description, ingredient, portion_options)
                response = llm.chat(prompt, image_data)

                # Parse response
                lines = response.strip().split('\n')
                if len(lines) >= 2 and lines[0] != "unknown":
                    amount = lines[0].strip()
                    unit = lines[1].strip()

                    # Extract base unit (remove leading "1 " if present)
                    # e.g., "1 cup" -> "cup", "1 oz" -> "oz"
                    base_unit = unit
                    if unit.startswith('1 '):
                        base_unit = unit[2:]  # Remove "1 "

                    # Calculate total amount
                    try:
                        amount_float = float(amount)
                        # If unit was "1 cup" and amount is 2, output "2 cup"
                        # If unit was "1 cup" and amount is 0.5, output "0.5 cup"
                        portion_results.append(f"{ingredient}: {amount} {base_unit}")
                    except:
                        # If can't parse amount, keep original format
                        portion_results.append(f"{ingredient}: {amount} {unit}")
                else:
                    portion_results.append(f"{ingredient}: unknown")

                time.sleep(0.5)  # Small delay between ingredient estimations

            except Exception as e:
                portion_results.append(f"{ingredient}: unknown")

        # Format results
        gpt_amount = '\n'.join(portion_results)
        gpt_food_code = match_ingredients_to_codes(ingredients_list, food_codes_list)

        return index, dish_id, gpt_amount, gpt_food_code, None

    except Exception as e:
        return index, dish_id, None, None, str(e)


def estimate_ingredient_portions_parallel(df, image_base_path, results_path, checkpoint_path, portion_weights_df, model_name, max_workers=5):
    """
    Estimate per-ingredient portions using FNDDS standard units with parallel processing.
    """
    last_index = load_checkpoint(checkpoint_path)
    print(f"Starting parallel per-ingredient portion estimation from index {last_index}")
    print(f"Max workers: {max_workers}")

    # Prepare tasks
    tasks = []
    for i in range(len(df)):
        if i < last_index:
            continue

        # Skip if already processed
        if not pd.isna(df.loc[i, 'GPTAmount']):
            continue

        dish_id = df.loc[i, 'dish_id']
        food_description = df.loc[i, 'GPTFoodDescription']
        food_codes_str = df.loc[i, 'GPTFoodCode']

        # Skip if no valid data
        if pd.isna(food_description) or str(food_description).startswith("I can't"):
            continue

        # Parse ingredients from GPT's identification
        ingredients_list = parse_ingredients_from_gpt_description(food_description)
        food_codes_list = parse_food_codes(food_codes_str)

        if not ingredients_list:
            continue

        image_path = os.path.join(image_base_path, dish_id, 'rgb.png')
        if not os.path.exists(image_path):
            continue

        tasks.append((i, dish_id, image_path, food_description, ingredients_list, food_codes_list, portion_weights_df, model_name))

    print(f"Found {len(tasks)} dishes to process")

    if not tasks:
        print("No dishes to process!")
        return

    # Process in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_single_dish_portions, task): task for task in tasks}

        for future in as_completed(future_to_task):
            index, dish_id, gpt_amount, gpt_food_code, error = future.result()
            completed += 1

            if error:
                print(f"[{completed}/{len(tasks)}] Error at index {index} ({dish_id}): {error}")
                df.loc[index, 'GPTAmount'] = f"Error: {error}"
            else:
                print(f"[{completed}/{len(tasks)}] Completed index {index} ({dish_id})")
                df.loc[index, 'GPTAmount'] = gpt_amount
                df.loc[index, 'GPTFoodCode'] = gpt_food_code

            # Save progress periodically
            if completed % 10 == 0:
                df.to_csv(results_path, index=False)
                save_checkpoint(index + 1, checkpoint_path)
                print(f"Progress saved: {completed}/{len(tasks)} completed")

    # Final save
    df.to_csv(results_path, index=False)
    save_checkpoint(len(df), checkpoint_path)
    print(f"Parallel per-ingredient portion estimation complete! Processed {len(tasks)} dishes.")

def main(args):
    print("="*70)
    print("Nutrition5k Per-Ingredient Portion Estimation (FNDDS Units)")
    print("Output format compatible with nutrient_estimate_mix.py")
    print("="*70)

    # Load data
    print(f"\nLoading results from: {args.results_file}")
    df = pd.read_csv(args.results_file)

    # Load FoodWeights for portion lookup
    print(f"Loading FoodWeights from: {args.food_weights_file}")
    portion_weights_df = pd.read_csv(args.food_weights_file)
    print(f"Loaded {len(portion_weights_df)} portion weight records")

    # Initialize columns if needed (use object dtype for string data)
    if 'GPTAmount' not in df.columns:
        df['GPTAmount'] = pd.Series([np.nan] * len(df), dtype='object')

    print(f"Loaded {len(df)} dishes")

    # Use parallel processing by default
    print(f"\n=== Running Parallel Mode ({args.max_workers} workers) ===")
    estimate_ingredient_portions_parallel(
        df=df,
        image_base_path=args.image_base_path,
        results_path=args.results_file,
        checkpoint_path=args.checkpoint_file,
        portion_weights_df=portion_weights_df,
        model_name=args.model,
        max_workers=args.max_workers
    )

    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Results saved to: {args.results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-ingredient portion estimation for Nutrition5k dataset using FNDDS units")
    parser.add_argument("--results_file", required=True,
                        help="Path to results CSV (output from nutrition5k_food_code.py)")
    parser.add_argument("--image_base_path", required=True,
                        help="Base path to Nutrition5k images")
    parser.add_argument("--food_weights_file", required=True,
                        help="Path to FoodWeights.csv with FNDDS portions")
    parser.add_argument("--checkpoint_file", required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--model", default="gpt-5-mini",
                        help="Vision model to use (default: gpt-5-mini)")
    parser.add_argument("--max_workers", type=int, default=5,
                        help="Number of parallel workers for processing (default: 5)")

    args = parser.parse_args()
    main(args)
