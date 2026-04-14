import argparse
import pandas as pd
from fractions import Fraction
import numpy as np
import time
import inflect
import os
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

from chagApp_openai import Vision as OpenAIVision
from chatApp_gemini import Vision as GeminiVision

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def has_common_element(list1, list2):
    return list(set(list1).intersection(set(list2)))

def extract_unit(value):
    return ' '.join(value.split(' ', 1)[1:])

def extract_amount(value):
    return value.split(' ', 1)[0]

def parse_fraction(value_str):
    """Parse a fraction string that may be a simple fraction, mixed fraction, or decimal.

    Examples:
        "2" -> 2.0
        "1/2" -> 0.5
        "1-1/2" -> 1.5
        "2 1/4" -> 2.25
    """
    value_str = value_str.strip()

    # Check for mixed fraction with hyphen (e.g., "1-1/2")
    if '-' in value_str and '/' in value_str:
        parts = value_str.split('-')
        if parts[0].isdigit():
            whole = int(parts[0])
            frac = float(Fraction(parts[1]))
            return whole + frac
        # Hyphen is part of the unit (e.g., "1/2-cup"), treat first part as a simple fraction
        return float(Fraction(parts[0]))

    # Check for mixed fraction with space (e.g., "1 1/2")
    elif ' ' in value_str and '/' in value_str:
        parts = value_str.split(' ', 1)
        whole = int(parts[0])
        frac = float(Fraction(parts[1]))
        return whole + frac

    # Simple fraction or integer
    else:
        return float(Fraction(value_str))

def singularize(word, p):
    return p.singular_noun(word) or word


def singularize_phrase(phrase, p):
    phrase = str(phrase).strip()
    if not phrase:
        return phrase

    full = p.singular_noun(phrase)
    if full:
        return full

    parts = phrase.split()
    if len(parts) == 1:
        return singularize(parts[0], p)

    last = p.singular_noun(parts[-1]) or parts[-1]
    return " ".join(parts[:-1] + [last])


def normalize_standard_portion(portion_str, p):
    """Map raw portion strings to a standard '1 <unit>' descriptor."""
    if pd.isna(portion_str):
        return None

    portion_str = str(portion_str).strip()
    if not portion_str:
        return None

    lower = portion_str.lower()
    if lower.startswith("quantity not specified") or lower.startswith("guideline "):
        return None

    amount_token = extract_amount(portion_str).strip()
    try:
        parse_fraction(amount_token)
    except Exception:
        return None

    unit = extract_unit(portion_str).strip()
    if not unit:
        return None

    return f"1 {singularize_phrase(unit, p)}"


def is_valid_portion_target(value):
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    return text != "I can't help to analyze this image."


def normalize_model_text(value):
    text = str(value).strip().strip('`"\'')
    if text.endswith('.'):
        text = text[:-1].rstrip()
    return " ".join(text.split()).lower()


def split_portion_options(portion_data):
    return [opt.strip() for opt in str(portion_data).split(' ,') if opt.strip()]


def match_portion_option(response_text, portion_data):
    normalized_response = normalize_model_text(response_text)
    for option in split_portion_options(portion_data):
        if normalize_model_text(option) == normalized_response:
            return option
    return None


def format_numeric_value(value):
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6f}".rstrip('0').rstrip('.')


def parse_numeric_multiplier(response_text):
    cleaned = str(response_text).strip().strip('`"\'').replace(',', '')
    value = parse_fraction(cleaned)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Amount must be a non-negative number")
    return format_numeric_value(value)

def load_image_as_base64(path_or_url, timeout=10):
    """Load image from a local file path or URL and convert to base64 data URL.

    Args:
        path_or_url: Local file path or image URL
        timeout: Request timeout in seconds (for URLs only)

    Returns:
        Base64 data URL string, or None if loading fails
    """
    try:
        # Check if it's a local file
        if os.path.isfile(path_or_url):
            ext = os.path.splitext(path_or_url)[1].lower()
            image_type = 'jpeg' if ext in ('.jpg', '.jpeg') else 'png'
            with open(path_or_url, 'rb') as f:
                encoded_string = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/{image_type};base64,{encoded_string}"

        # Otherwise treat as URL
        response = requests.get(path_or_url, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if 'image/jpeg' in content_type or path_or_url.lower().endswith(('.jpg', '.jpeg')):
            image_type = 'jpeg'
        elif 'image/png' in content_type or path_or_url.lower().endswith('.png'):
            image_type = 'png'
        else:
            image_type = 'png'

        encoded_string = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/{image_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Failed to load image from {path_or_url}: {e}")
        return None

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


def create_vision_client(provider, model_name):
    if provider == 'openai':
        return OpenAIVision(model_name)
    if provider == 'gemini':
        return GeminiVision(model_name)
    raise ValueError(f"Unsupported provider: {provider}")

def create_prompt(food_description, portion_data, type='shot'):
    if type == 'shot':
        return f"""The image displays the food described as {food_description}. Please assess the portion size shown in the image and 
choose the option that most accurately represents it from the following choices: {portion_data}. The selected portion size will be used as the reference portion for counting in the next step. Respond with only the selected portion size. 
If you cannot analyze the image, reply with "I can't help to analyze this image." and specify the reason on a new line."""
    else:
        return f"""The image displays the food described as {food_description}. Please evaluate the portion size presented and specify the quantity: 
how many {portion_data} of the food are visible in the image? Please respond only with the numerical value. 
If you cannot analyze the image, reply with "I can't help to analyze this image." and specify the reason on a new line."""

def update_dataframe(df, index, response, column_desc, column_reason, portion_data, type='shot'):
    response_splits = response.split('\n')
    df[column_desc] = df[column_desc].astype(object)
    df[column_reason] = df[column_reason].astype(object)
    if response_splits[0] == "I can't help to analyze this image.":
        df.loc[index, column_desc] = response_splits[0]
        df.loc[index, column_reason] = response_splits[1] if len(response_splits) > 1 else np.nan
    else:
        first_line = response_splits[0].strip()
        if type == 'shot':
            matched_option = match_portion_option(first_line, portion_data)
            if matched_option is None:
                df.loc[index, column_desc] = np.nan
                df.loc[index, column_reason] = (
                    f'Invalid portion selection (must match one listed option): {first_line}'
                )
            else:
                df.loc[index, column_desc] = matched_option
                df.loc[index, column_reason] = np.nan
        else:
            try:
                df.loc[index, column_desc] = parse_numeric_multiplier(first_line)
                df.loc[index, column_reason] = np.nan
            except Exception:
                df.loc[index, column_desc] = np.nan
                df.loc[index, column_reason] = (
                    f'Invalid numeric multiplier response: {first_line}'
                )

def process_dataframe(df_results, df_image_link, p):
    food_code_col = 'FoodCodeCommon' if 'FoodCodeCommon' in df_results.columns else 'FoodCode'
    ls_label_amount = []
    ls_label_unit = []
    ls_portion_shot = []
    for i in range(len(df_results)):
        str_portion = df_results.loc[i, 'Portion']
        str_food_code = df_results.loc[i, food_code_col]

        # Handle missing portion values
        if pd.isna(str_portion):
            ls_label_amount.append(np.nan)
            ls_label_unit.append(np.nan)
            ls_portion_shot.append('')
            continue

        ls_label_amount.append(parse_fraction(extract_amount(str_portion)))
        ls_label_unit.append(singularize(extract_unit(str_portion), p))
        df_image_link_sel = df_image_link[df_image_link['FoodCode'] == float(str_food_code)].copy()
        df_image_link_sel.sort_values(by=['Multiplier'], ascending=False, inplace=True)
        raw_portions = df_image_link_sel['Portion'].dropna().astype(str).tolist()
        normalized_portions = []
        seen = set()
        for portion in raw_portions:
            normalized = normalize_standard_portion(portion, p)
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_portions.append(normalized)
        ls_portion_shot.append(' ,'.join(normalized_portions))
    df_results['LabelAmount'] = ls_label_amount
    df_results['LabelUnit'] = ls_label_unit
    df_results['PortionShot'] = ls_portion_shot
    return df_results

def analyze_portions(df, llm, column_desc, column_reason, portion_data, checkpoint_path,
                     output_path, type='shot'):
    """Analyze portions with checkpointing support for resume capability."""
    # Load checkpoint to resume from last processed index
    last_index = load_checkpoint(checkpoint_path)
    print(f"Starting from index {last_index} (checkpoint: {checkpoint_path})")

    ls_url_str = df['Link'].tolist()
    for i, url_str in enumerate(ls_url_str):
        # Skip already processed rows
        if i < last_index:
            continue

        # Skip if already has a valid result (allows manual reruns)
        if not pd.isna(df.loc[i, column_desc]):
            print(f"Skipping index {i} - already processed")
            save_checkpoint(i + 1, checkpoint_path)
            continue

        if not pd.isna(df.loc[i, 'GPTFoodDescription']):
            food_description = df.loc[i, 'GPTFoodDescription']
            print(f"Processing index {i}/{len(df)}: {food_description[:50]}...")
            try:
                # Download image and convert to base64
                image_data = load_image_as_base64(url_str)
                if image_data is None:
                    print(f"Skipping index {i} - could not download image")
                    save_checkpoint(i + 1, checkpoint_path)
                    continue

                portion_target = portion_data[i]
                if not is_valid_portion_target(portion_target):
                    print(f"Skipping index {i} - no valid portion target")
                    save_checkpoint(i + 1, checkpoint_path)
                    continue

                prompt = create_prompt(food_description, portion_target, type)
                response = llm.chat(prompt, image_data)
                update_dataframe(df, i, response, column_desc, column_reason, portion_target, type=type)
                df.to_csv(output_path, index=False)
                save_checkpoint(i + 1, checkpoint_path)
                time.sleep(2)  # Reduced from 10s to 2s - adjust if you hit rate limits
            except Exception as e:
                print(f"Error at index {i}: {e}")
                continue
        else:
            print(f"Skipping index {i} - no GPTFoodDescription")
            save_checkpoint(i + 1, checkpoint_path)

    print(f"Analysis complete! Processed {len(df)} rows.")

def process_single_image(args):
    """Process a single image - used for parallel processing."""
    i, url_str, food_description, portion_data_item, type_mode, provider, model_name = args

    try:
        # Download image and convert to base64
        image_data = load_image_as_base64(url_str)
        if image_data is None:
            return i, None, f"Could not download image from {url_str}"

        # Create LLM instance for this thread
        llm = create_vision_client(provider, model_name)

        prompt = create_prompt(food_description, portion_data_item, type_mode)
        response = llm.chat(prompt, image_data)
        return i, response, None
    except Exception as e:
        return i, None, str(e)

def analyze_portions_parallel(df, column_desc, column_reason, portion_data, checkpoint_path,
                              output_path, provider='openai', model_name='gpt-5-mini',
                              type='shot', max_workers=5):
    """Analyze portions with parallel processing for much faster execution.

    Args:
        max_workers: Number of concurrent threads (default 5, adjust based on rate limits)
    """
    last_index = load_checkpoint(checkpoint_path)
    print(f"Starting parallel processing from index {last_index} with {max_workers} workers")
    print(f"Checkpoint: {checkpoint_path}")

    # Prepare tasks for parallel processing
    tasks = []
    for i in range(len(df)):
        if i < last_index:
            continue

        if not pd.isna(df.loc[i, column_desc]):
            continue

        if pd.isna(df.loc[i, 'GPTFoodDescription']):
            continue

        url_str = df.loc[i, 'Link']
        food_description = df.loc[i, 'GPTFoodDescription']
        portion_target = portion_data[i]
        if not is_valid_portion_target(portion_target):
            continue
        tasks.append((i, url_str, food_description, portion_target, type, provider, model_name))

    print(f"Found {len(tasks)} images to process")

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_single_image, task): task[0] for task in tasks}

        completed = 0
        for future in as_completed(future_to_index):
            i, response, error = future.result()
            completed += 1

            if error:
                print(f"[{completed}/{len(tasks)}] Error at index {i}: {error}")
                df.loc[i, column_reason] = error
            else:
                update_dataframe(df, i, response, column_desc, column_reason, portion_data[i], type=type)
                print(f"[{completed}/{len(tasks)}] Completed index {i}")

            # Save progress periodically (every 10 images)
            if completed % 10 == 0:
                df.to_csv(output_path, index=False)
                save_checkpoint(i + 1, checkpoint_path)
                print(f"Checkpoint saved at {completed} completed")

    # Final save
    df.to_csv(output_path, index=False)
    print(f"Parallel analysis complete! Processed {len(tasks)} images.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run portion-size estimation with OpenAI or Gemini vision models.")
    parser.add_argument('--input', type=str, default='../data/ASA24_GPTFoodCodes_all_portions.csv',
                        help='Input CSV containing Link, Portion, GPTFoodDescription, and FoodCode or FoodCodeCommon')
    parser.add_argument('--image_metadata', type=str, default='../data/ASA24_ImageMetadata.csv',
                        help='CSV with FoodCode-to-Portion mappings used to build portion choices')
    parser.add_argument('--output', type=str, default='../data/ASA24_GPTFoodCodes_portion.csv',
                        help='Output CSV path')
    parser.add_argument('--checkpoint_description', type=str,
                        default='checkpoints/checkpoint_portion_description.txt',
                        help='Checkpoint path for portion-description prompts')
    parser.add_argument('--checkpoint_amount', type=str,
                        default='checkpoints/checkpoint_portion_amount.txt',
                        help='Checkpoint path for portion-amount prompts')
    parser.add_argument('--provider', type=str, choices=['openai', 'gemini'],
                        default='openai',
                        help='Vision provider to use (default: openai)')
    parser.add_argument('--model', type=str, default='gpt-5-mini',
                        help='Vision model name for the selected provider')
    parser.add_argument('--max_workers', type=int, default=5,
                        help='Number of concurrent workers for parallel mode')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential processing instead of parallel processing')
    parser.add_argument('--old_file_filter', type=str,
                        default='../data/archive/ASA24_GPTFoodCodes_metrics_final_old.csv',
                        help='Optional CSV whose FileName set is used to filter inputs; pass empty string to disable')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load data
    df_results = pd.read_csv(args.input)
    if args.old_file_filter:
        old_filter_df = pd.read_csv(args.old_file_filter)
        old_fns = set(old_filter_df['FileName'].unique())
        df_results = df_results[df_results['FileName'].isin(old_fns)].reset_index(drop=True)
    df_image_link = pd.read_csv(args.image_metadata)

    # Initialize inflect engine
    p = inflect.engine()

    # Process dataframe - adds LabelAmount, LabelUnit, PortionShot columns
    df_results = process_dataframe(df_results, df_image_link, p)

    # Choose processing mode: parallel (faster) or sequential (safer for rate limits)
    use_parallel = not args.sequential

    if use_parallel:
        print("\n=== Using PARALLEL processing mode ===")
        print(f"Max workers: {args.max_workers}")
        print("This is much faster but may hit rate limits. Reduce MAX_WORKERS if needed.\n")

        # Analyze portions - Description mode
        print("\n=== Starting Portion Description Analysis (Parallel) ===")
        if 'GPTPortionDescription' not in df_results.columns:
            df_results['GPTPortionDescription'] = [np.nan] * len(df_results)
            df_results['GPTPortionReason'] = [np.nan] * len(df_results)
        analyze_portions_parallel(df_results, 'GPTPortionDescription', 'GPTPortionReason',
                                 df_results['PortionShot'], args.checkpoint_description,
                                 args.output, provider=args.provider, model_name=args.model,
                                 type='shot', max_workers=args.max_workers)

        # Analyze portions - Amount mode
        print("\n=== Starting Portion Amount Analysis (Parallel) ===")
        if 'GPTPortionAmount' not in df_results.columns:
            df_results['GPTPortionAmount'] = [np.nan] * len(df_results)
            df_results['GPTPortionAmountReason'] = [np.nan] * len(df_results)
        analyze_portions_parallel(df_results, 'GPTPortionAmount', 'GPTPortionAmountReason',
                                 df_results['GPTPortionDescription'], args.checkpoint_amount,
                                 args.output, provider=args.provider, model_name=args.model,
                                 type='amount', max_workers=args.max_workers)
    else:
        print("\n=== Using SEQUENTIAL processing mode ===")
        print("Slower but safer for rate limits.\n")

        # Initialize Vision API
        llm = create_vision_client(args.provider, args.model)

        # Analyze portions - Description mode
        print("\n=== Starting Portion Description Analysis ===")
        if 'GPTPortionDescription' not in df_results.columns:
            df_results['GPTPortionDescription'] = [np.nan] * len(df_results)
            df_results['GPTPortionReason'] = [np.nan] * len(df_results)
        analyze_portions(df_results, llm, 'GPTPortionDescription', 'GPTPortionReason',
                        df_results['PortionShot'], args.checkpoint_description,
                        args.output, type='shot')

        # Clear message history to prevent token overflow in second analysis
        print("\nClearing message history before starting amount analysis...")
        llm.messages = [llm.messages[0]]  # Keep only system prompt

        # Analyze portions - Amount mode
        print("\n=== Starting Portion Amount Analysis ===")
        if 'GPTPortionAmount' not in df_results.columns:
            df_results['GPTPortionAmount'] = [np.nan] * len(df_results)
            df_results['GPTPortionAmountReason'] = [np.nan] * len(df_results)
        analyze_portions(df_results, llm, 'GPTPortionAmount', 'GPTPortionAmountReason',
                        df_results['GPTPortionDescription'], args.checkpoint_amount,
                        args.output, type='amount')

    print("\n=== All portion analysis complete! ===")

if __name__ == "__main__":
    main()
