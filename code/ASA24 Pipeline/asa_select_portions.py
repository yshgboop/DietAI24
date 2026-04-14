"""
ASA24 Portion Selection Script

This script selects largest, smallest, and median portions for each of the top 1000
most frequently consumed foods from the ASA24 dataset. It generates a single CSV file
with 3000 rows (3 portions × 1000 foods) labeled with a PortionType column
(largest/median/smallest) for use with the ViT baseline and other analyses.

Author: DietAI24 Project
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
import sys


def setup_logging(log_file):
    """
    Configure logging to write to both file and console.

    Args:
        log_file (str): Path to log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_data(top_foods_path, image_link_path, image_dir):
    """
    Load and validate input CSV files, constructing local image paths.

    Args:
        top_foods_path (str): Path to top_1000_frequent_select_foods.csv
        image_link_path (str): Path to df_image_link.csv
        image_dir (str): Local directory containing the portion images

    Returns:
        tuple: (df_top_foods, df_image_link with Link column as local paths)

    Raises:
        FileNotFoundError: If input files don't exist
        KeyError: If required columns are missing
        ValueError: If data is malformed
    """
    logging.info("Loading input data files...")

    # Load top 1000 foods
    try:
        df_top_foods = pd.read_csv(top_foods_path)
        logging.info(f"Loaded top foods CSV: {df_top_foods.shape[0]} rows")
    except FileNotFoundError:
        logging.error(f"File not found: {top_foods_path}")
        raise

    # Validate top_foods columns
    required_top_foods_cols = ['Food code', 'Main Food description']
    missing_cols = [col for col in required_top_foods_cols if col not in df_top_foods.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in top foods CSV: {missing_cols}. "
                      f"Available columns: {list(df_top_foods.columns)}")

    # Load image link metadata
    try:
        df_image_link = pd.read_csv(image_link_path)
        logging.info(f"Loaded image link CSV: {df_image_link.shape[0]} rows")
    except FileNotFoundError:
        logging.error(f"File not found: {image_link_path}")
        raise

    # Validate image_link columns
    required_image_link_cols = ['FoodCode', 'Portion', 'Multiplier']
    missing_cols = [col for col in required_image_link_cols if col not in df_image_link.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in image link CSV: {missing_cols}. "
                      f"Available columns: {list(df_image_link.columns)}")

    # Convert food codes to 8-digit strings with zero-padding
    # Handle potential float types by converting to int first (removes .0 decimal)
    # First, drop rows with missing FoodCode values
    original_image_link_count = len(df_image_link)
    df_image_link = df_image_link[df_image_link['FoodCode'].notna()].copy()
    dropped_count = original_image_link_count - len(df_image_link)
    if dropped_count > 0:
        logging.warning(f"Dropped {dropped_count} rows from image_link with missing FoodCode")

    df_top_foods['Food code'] = df_top_foods['Food code'].astype(float).astype(int).astype(str).str.strip().str.zfill(8)
    df_image_link['FoodCode'] = df_image_link['FoodCode'].astype(float).astype(int).astype(str).str.strip().str.zfill(8)

    # Try to convert Multiplier to float
    try:
        df_image_link['Multiplier'] = pd.to_numeric(df_image_link['Multiplier'], errors='coerce')
        null_multipliers = df_image_link['Multiplier'].isna().sum()
        if null_multipliers > 0:
            logging.warning(f"Found {null_multipliers} non-numeric Multiplier values, set to NaN")
    except Exception as e:
        logging.error(f"Error converting Multiplier to numeric: {e}")
        raise ValueError("Multiplier column must contain numeric values")

    # Check for empty dataframes
    if len(df_top_foods) == 0:
        raise ValueError("Top foods CSV is empty")
    if len(df_image_link) == 0:
        raise ValueError("Image link CSV is empty")

    # Construct local image paths from FileName and image directory
    logging.info(f"Constructing local image paths from directory: {image_dir}")
    df_image_link['Link'] = df_image_link['FileName'].apply(
        lambda f: os.path.join(image_dir, f) if pd.notna(f) else None
    )

    # Verify a sample of paths exist
    sample_paths = df_image_link['Link'].dropna().head(5).tolist()
    missing_sample = [p for p in sample_paths if not os.path.exists(p)]
    if missing_sample:
        logging.warning(f"Some image files not found locally, e.g.: {missing_sample[0]}")
    else:
        logging.info(f"Sample image paths verified OK (checked {len(sample_paths)} files)")

    logging.info(f"Data loading complete. Top foods: {len(df_top_foods)}, Image links: {len(df_image_link)}")
    logging.info(f"Sample food codes: {df_top_foods['Food code'].head(3).tolist()}")

    # Add debug logging for food codes
    logging.info("\n" + "=" * 50)
    logging.info("FOOD CODE DEBUG INFO")
    logging.info("=" * 50)
    logging.info(f"Top foods - Sample food codes: {df_top_foods['Food code'].head(5).tolist()}")
    logging.info(f"Top foods - Food code dtype: {df_top_foods['Food code'].dtype}")
    logging.info(f"Image link - Sample food codes: {df_image_link['FoodCode'].head(5).tolist()}")
    logging.info(f"Image link - Food code dtype: {df_image_link['FoodCode'].dtype}")
    logging.info(f"Image link - Total unique food codes: {df_image_link['FoodCode'].nunique()}")

    # Check for overlap
    top_codes_set = set(df_top_foods['Food code'].astype(str))
    image_codes_set = set(df_image_link['FoodCode'].astype(str))
    overlap = top_codes_set & image_codes_set
    logging.info(f"Overlapping food codes: {len(overlap)} out of {len(top_codes_set)}")
    if len(overlap) > 0:
        logging.info(f"Sample overlapping codes: {list(overlap)[:5]}")
    else:
        logging.warning("NO OVERLAP FOUND! Checking first few codes from each file:")
        logging.info(f"  First 3 from top_foods: {list(top_codes_set)[:3]}")
        logging.info(f"  First 3 from image_link: {list(image_codes_set)[:3]}")
    logging.info("=" * 50 + "\n")

    return df_top_foods, df_image_link


def select_portions_for_food(food_code, df_image_link):
    """
    Select three portions (largest, smallest, median) for a single food code.

    Based on the logic in code/rag_portion_size.py:56-58, this function:
    1. Filters df_image_link for the specific food code
    2. Sorts by Multiplier in descending order (largest first)
    3. Selects largest (first), smallest (last), and median (middle)

    Args:
        food_code (str): 8-digit FNDDS food code
        df_image_link (pd.DataFrame): DataFrame with portion metadata

    Returns:
        tuple: (dict of portions, status message)
            - dict has keys: 'largest', 'median', 'smallest'
            - Each value is a pandas Series (row from df_image_link)
            - Returns (None, error_message) if no portions found
    """
    # Filter for specific food code
    df_portions = df_image_link[df_image_link['FoodCode'] == food_code].copy()

    # Remove rows with NaN multipliers
    df_portions = df_portions[df_portions['Multiplier'].notna()]

    n_portions = len(df_portions)

    if n_portions == 0:
        return None, "No portions found"

    # Sort by Multiplier descending (largest first)
    df_sorted = df_portions.sort_values('Multiplier', ascending=False, ignore_index=True)

    if n_portions == 1:
        # Only one portion available — no duplication
        row = df_sorted.iloc[0]
        return {
            'largest': row.copy(),
        }, "Only 1 portion"

    elif n_portions == 2:
        # Two distinct portions — largest and smallest only
        return {
            'largest': df_sorted.iloc[0].copy(),
            'smallest': df_sorted.iloc[1].copy(),
        }, "Only 2 portions"

    else:
        # Normal case: 3+ portions available
        # For median: use lower-middle index (n//2) for even-length lists
        median_idx = n_portions // 2
        return {
            'largest': df_sorted.iloc[0].copy(),
            'median': df_sorted.iloc[median_idx].copy(),
            'smallest': df_sorted.iloc[-1].copy()
        }, "Complete"


def process_food_selections(df_top_foods, df_image_link):
    """
    Main processing loop to select portions for all 1000 foods.

    Args:
        df_top_foods (pd.DataFrame): Top 1000 frequent foods
        df_image_link (pd.DataFrame): Image link metadata with portions

    Returns:
        tuple: (results_list, stats_dict)
            - results_list: List of pandas Series, each with PortionType set
            - stats_dict: Statistics about the selection process
    """
    logging.info("Starting food selection process...")

    results = []
    stats = {
        'complete': 0,
        'partial': 0,
        'failed': 0,
        'missing_foods': [],
        'portion_counts': {}
    }

    total_foods = len(df_top_foods)

    for idx, row in df_top_foods.iterrows():
        food_code = row['Food code']
        food_desc = row['Main Food description']
        rank = idx

        # Progress logging every 100 foods
        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1}/{total_foods} foods...")

        # Select portions for this food
        portions, status = select_portions_for_food(food_code, df_image_link)

        if portions is None:
            stats['failed'] += 1
            stats['missing_foods'].append(food_code)
            logging.warning(f"Food {food_code} ({food_desc}): {status}")
            continue

        # Track statistics
        if status == "Complete":
            stats['complete'] += 1
        else:
            stats['partial'] += 1
            logging.info(f"Food {food_code} ({food_desc}): {status}")

        # Count number of available portions
        n_portions = len(df_image_link[df_image_link['FoodCode'] == food_code])
        stats['portion_counts'][n_portions] = stats['portion_counts'].get(n_portions, 0) + 1

        # Build output rows only for portion types that exist (no duplication)
        for size_key, portion_series in portions.items():
            portion_row = portion_series.copy()

            # Add food metadata
            portion_row['FoodCode'] = food_code
            portion_row['Main Food description'] = food_desc
            portion_row['Rank'] = rank
            portion_row['PortionType'] = size_key

            results.append(portion_row)

    logging.info(f"Selection process complete. Processed {total_foods} foods.")

    return results, stats


def save_outputs(results, output_dir):
    """
    Save a single CSV file with all selected portions and a PortionType label column.

    Args:
        results (list): List of pandas Series, each with a 'PortionType' field
        output_dir (str): Directory to save output file

    Returns:
        str: Path to saved file
    """
    logging.info("Saving output CSV file...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert list of Series to DataFrame (PortionType already set per row)
    df_output = pd.DataFrame(results)

    # Define column order (FileName first for compatibility, PortionType second for clarity)
    # Include PortionCode and PortionSubCode for nutrient estimation
    column_order = ['FileName', 'PortionType', 'FoodCode', 'Main Food description', 'Portion', 'Multiplier', 'Rank', 'PortionCode', 'PortionSubCode']

    # Check if 'Link' column exists, if so add it to column order
    if 'Link' in df_output.columns:
        if 'Link' not in column_order:
            column_order.append('Link')

    # Reorder columns (only include columns that exist)
    available_columns = [col for col in column_order if col in df_output.columns]
    df_output = df_output[available_columns]

    # Save to CSV
    output_file_name = 'ASA24_GPTFoodCodes_all_portions.csv'
    output_file_path = output_path / output_file_name
    df_output.to_csv(output_file_path, index=False)

    logging.info(f"Created: {output_file_path} ({len(df_output)} rows)")
    portion_counts = df_output['PortionType'].value_counts()
    for ptype in ['largest', 'median', 'smallest']:
        count = portion_counts.get(ptype, 0)
        logging.info(f"  - {ptype.capitalize()} portions: {count} rows")

    return str(output_file_path)


def generate_statistics(stats):
    """
    Generate and print summary statistics report.

    Args:
        stats (dict): Statistics dictionary from process_food_selections
    """
    logging.info("\n" + "=" * 50)
    logging.info("SELECTION STATISTICS")
    logging.info("=" * 50)

    total_processed = stats['complete'] + stats['partial'] + stats['failed']
    logging.info(f"Total foods processed: {total_processed}")
    logging.info(f"Foods with complete selections (3+ portions): {stats['complete']}")
    logging.info(f"Foods with partial selections (1-2 portions): {stats['partial']}")
    logging.info(f"Foods with no portions found: {stats['failed']}")

    logging.info("\nPortion availability distribution:")
    for n_portions in sorted(stats['portion_counts'].keys()):
        count = stats['portion_counts'][n_portions]
        logging.info(f"  {n_portions} portion(s): {count} foods")

    if stats['missing_foods']:
        logging.info(f"\nMissing food codes ({len(stats['missing_foods'])} total):")
        # Show first 10
        for food_code in stats['missing_foods'][:10]:
            logging.info(f"  {food_code}")
        if len(stats['missing_foods']) > 10:
            logging.info(f"  ... and {len(stats['missing_foods']) - 10} more")

    logging.info("=" * 50 + "\n")


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Select largest, smallest, and median portions for ASA24 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python code/asa_select_portions.py \\
    --image_link_csv /path/to/df_image_link.csv \\
    --link_file /path/to/file_with_links.csv

  python code/asa_select_portions.py \\
    --image_link_csv ../df_image_link.csv \\
    --link_file ../asa24_image_links.csv \\
    --image_dir ../ASA24_images/ \\
    --validate_images
        """
    )

    parser.add_argument(
        '--top_foods_csv',
        type=str,
        default='../NHANES/top_1000_frequent_select_foods.csv',
        help='Path to top 1000 frequent foods CSV (default: ../NHANES/top_1000_frequent_select_foods.csv)'
    )

    parser.add_argument(
        '--image_link_csv',
        type=str,
        default='/Volumes/My Passport/Asa24PortionImages/ASA24_ImageMetadata.csv',
        help='Path to df_image_link.csv with ASA24 portion metadata (required)'
    )

    parser.add_argument(
        '--image_dir',
        type=str,
        default='/Volumes/My Passport/Asa24PortionImages/Asa24PortionImages',
        help='Local directory containing the ASA24 portion images'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../',
        help='Directory to save output CSV files (default: ../)'
    )

    parser.add_argument(
        '--log_file',
        type=str,
        default='asa_select_portions.log',
        help='Path to log file (default: asa_select_portions.log)'
    )

    return parser.parse_args()


def main():
    """
    Main function to orchestrate the portion selection process.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_file)

    logging.info("=" * 50)
    logging.info("ASA24 Portion Selection Script")
    logging.info("=" * 50)
    logging.info(f"Top foods CSV: {args.top_foods_csv}")
    logging.info(f"Image link CSV: {args.image_link_csv}")
    logging.info(f"Image directory: {args.image_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Log file: {args.log_file}")
    logging.info("=" * 50 + "\n")

    try:
        # Step 1: Load data
        df_top_foods, df_image_link = load_data(args.top_foods_csv, args.image_link_csv, args.image_dir)

        # Step 2: Process food selections
        results, stats = process_food_selections(df_top_foods, df_image_link)

        # Step 3: Save outputs
        output_file_path = save_outputs(results, args.output_dir)

        # Step 4: Generate statistics
        generate_statistics(stats)


        logging.info("\n" + "=" * 50)
        logging.info("OUTPUT FILE CREATED")
        logging.info("=" * 50)
        logging.info(f"File: {output_file_path}")
        logging.info(f"Total rows: {len(results)}")
        logging.info("=" * 50 + "\n")

        logging.info("Script completed successfully!")

    except Exception as e:
        logging.error(f"Script failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
