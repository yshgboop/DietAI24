"""
Food Code and Portion Size Evaluation Script for ASA24 Dataset

Evaluates GPT-based predictions using:

1. Food Code Inference (hierarchical matching metrics):
   - Exact Match: All 8 digits match
   - Close Match: First 2 digits match (Subgroup)
   - Far Match: First 1 digit matches (Major Group)
   - Mismatch: First digit differs
   - No Prediction: GPTFoodCode is null/empty

2. Portion Size Weight Error:
   - Converts both ground truth and GPT-predicted portions to grams via FoodWeights.csv
   - GT weight  = weight_per_unit("1 {LabelUnit}") × LabelAmount
   - GPT weight = weight_per_unit(GPTPortionDescription) × GPTPortionAmount
   - Reports MAE (g), MAPE (%), Median AE (g) broken down by PortionType
"""

import pandas as pd
import numpy as np
import ast
import argparse
from collections import defaultdict


# FNDDS Major Food Groups (first digit)
FOOD_GROUPS = {
    '1': 'Milk and Dairy',
    '2': 'Meat, Poultry, Fish',
    '3': 'Eggs',
    '4': 'Legumes, Nuts, Seeds',
    '5': 'Grain Products',
    '6': 'Fruits',
    '7': 'Vegetables',
    '8': 'Fats, Oils, Salad Dressings',
    '9': 'Sugars, Sweets, Beverages',
}

# Responses that indicate no valid portion was estimated
NO_PREDICTION_PHRASES = ["quantity not specified", "i can't help"]


def parse_gpt_food_code(code_str):
    """
    Parse GPTFoodCode from string list format to 8-digit string.

    Args:
        code_str: String like "['74401010']" or NaN/empty

    Returns:
        8-digit string or None if invalid/empty
    """
    if pd.isna(code_str) or code_str == '' or code_str == '[]':
        return None

    try:
        # Parse the string list format
        code_list = ast.literal_eval(code_str)
        if isinstance(code_list, list) and len(code_list) > 0:
            code = str(code_list[0])
            # Ensure it's 8 digits (pad with leading zeros if needed)
            if code.isdigit():
                return code.zfill(8)
        return None
    except (ValueError, SyntaxError):
        # Try direct string conversion
        code_str = str(code_str).strip()
        if code_str.isdigit():
            return code_str.zfill(8)
        return None


def classify_match(gt_code, pred_code):
    """
    Classify the match type between ground truth and predicted code.

    Args:
        gt_code: Ground truth food code (int or string)
        pred_code: Predicted food code (string) or None

    Returns:
        str: 'exact', 'close', 'far', 'mismatch', or 'no_prediction'
    """
    if pred_code is None:
        return 'no_prediction'

    # Convert ground truth to 8-digit string
    gt_str = str(int(gt_code)).zfill(8)
    pred_str = pred_code.zfill(8)

    # Exact match: all 8 digits match
    if gt_str == pred_str:
        return 'exact'

    # Close match: first 2 digits match (same subgroup)
    if gt_str[:2] == pred_str[:2]:
        return 'close'

    # Far match: first 1 digit matches (same major group)
    if gt_str[0] == pred_str[0]:
        return 'far'

    # Mismatch: first digit differs
    return 'mismatch'


def lookup_portion_weight(food_code, portion_desc, df_weights):
    """
    Look up the weight (g) for a given food code and portion descriptor.

    Args:
        food_code: FNDDS food code (int)
        portion_desc: Portion descriptor string (e.g., "1 cup", "1 FO")
        df_weights: FoodWeights DataFrame with columns FoodCode, Portion, Portion weight (g)

    Returns:
        float weight in grams, or None if not found
    """
    if pd.isna(food_code) or pd.isna(portion_desc) or portion_desc == '':
        return None

    desc_lower = str(portion_desc).strip().lower()
    subset = df_weights[df_weights['FoodCode'] == int(food_code)]
    match = subset[subset['Portion'].str.strip().str.lower() == desc_lower]

    if len(match) == 0:
        return None
    return float(match.iloc[0]['Portion weight (g)'])


def evaluate_portion_weight(df, df_weights):
    """
    Evaluate portion size accuracy by comparing GT and GPT-estimated food weights.

    GT weight  = weight_per("1 {LabelUnit}") × LabelAmount
    GPT weight = weight_per(GPTPortionDescription) × GPTPortionAmount

    Args:
        df: DataFrame with portion columns
        df_weights: FoodWeights DataFrame

    Returns:
        dict with df_with_weights and aggregate error metrics
    """
    df_eval = df.copy()
    gt_weights = []
    gpt_weights = []

    for _, row in df_eval.iterrows():
        food_code = row.get('FoodCode')

        # --- Ground truth weight ---
        label_amount = row.get('LabelAmount')
        label_unit = row.get('LabelUnit')
        if pd.notna(label_amount) and pd.notna(label_unit) and str(label_unit).strip() != '':
            gt_desc = f"1 {str(label_unit).strip()}"
            w = lookup_portion_weight(food_code, gt_desc, df_weights)
            gt_w = float(label_amount) * w if w is not None else None
        else:
            gt_w = None
        gt_weights.append(gt_w)

        # --- GPT estimated weight ---
        gpt_desc = row.get('GPTPortionDescription')
        gpt_amount = row.get('GPTPortionAmount')

        # Treat non-answer responses as no prediction
        is_no_pred = (
            pd.isna(gpt_desc) or str(gpt_desc).strip() == '' or
            any(p in str(gpt_desc).strip().lower() for p in NO_PREDICTION_PHRASES)
        )
        try:
            gpt_amount_float = None if pd.isna(gpt_amount) else float(gpt_amount)
        except (ValueError, TypeError):
            gpt_amount_float = None

        if is_no_pred or gpt_amount_float is None:
            gpt_w = None
        else:
            w = lookup_portion_weight(food_code, gpt_desc, df_weights)
            gpt_w = gpt_amount_float * w if w is not None else None
        gpt_weights.append(gpt_w)

    df_eval['GTWeight'] = gt_weights
    df_eval['GPTWeight'] = gpt_weights

    # Compute per-row errors where both weights are available and GT > 0
    abs_errors = []
    pct_errors = []
    for gt_w, gpt_w in zip(gt_weights, gpt_weights):
        if gt_w is not None and gpt_w is not None and gt_w > 0:
            ae = abs(gt_w - gpt_w)
            abs_errors.append(ae)
            pct_errors.append(ae / gt_w * 100)
        else:
            abs_errors.append(None)
            pct_errors.append(None)

    df_eval['AbsError'] = abs_errors
    df_eval['PctError'] = pct_errors

    valid_ae = [x for x in abs_errors if x is not None]
    valid_pe = [x for x in pct_errors if x is not None]
    no_pred = sum(1 for g in gpt_weights if g is None)
    no_gt = sum(1 for g in gt_weights if g is None)

    return {
        'df_with_weights': df_eval,
        'mae': float(np.mean(valid_ae)) if valid_ae else None,
        'mape': float(np.mean(valid_pe)) if valid_pe else None,
        'median_ae': float(np.median(valid_ae)) if valid_ae else None,
        'no_prediction': no_pred,
        'no_gt': no_gt,
        'n_valid': len(valid_ae),
    }


def evaluate_food_codes(df, filter_valid=True):
    """
    Main evaluation function for food code predictions.

    Args:
        df: DataFrame with FoodCode and GPTFoodCode columns
        filter_valid: If True, exclude rows without FoodCode

    Returns:
        dict: Evaluation metrics with counts and percentages
    """
    # Filter to valid rows (those with ground truth food codes)
    if filter_valid:
        df_eval = df[df['FoodCode'].notna()].copy()
    else:
        df_eval = df.copy()

    total_samples = len(df_eval)

    # Initialize counters
    match_counts = defaultdict(int)
    group_stats = defaultdict(lambda: defaultdict(int))

    # Classify each row
    classifications = []
    for _, row in df_eval.iterrows():
        gt_code = row['FoodCode']
        pred_code = parse_gpt_food_code(row['GPTFoodCode'])

        match_type = classify_match(gt_code, pred_code)
        match_counts[match_type] += 1
        classifications.append(match_type)

        # Track by food group
        gt_str = str(int(gt_code)).zfill(8)
        group = gt_str[0]
        group_stats[group][match_type] += 1
        group_stats[group]['total'] += 1

    df_eval['MatchType'] = classifications

    # Calculate percentages
    metrics = {
        'total_samples': total_samples,
        'valid_predictions': total_samples - match_counts['no_prediction'],
        'counts': dict(match_counts),
        'percentages': {
            k: (v / total_samples * 100) if total_samples > 0 else 0
            for k, v in match_counts.items()
        },
        'group_stats': {k: dict(v) for k, v in group_stats.items()},
        'df_with_classifications': df_eval
    }

    return metrics


def compute_cumulative_metrics(df_subset):
    """
    Compute cumulative metrics for a subset of data.

    Returns dict with:
    - exact_pct: Exact match percentage
    - close_pct: Cumulative (exact + close) percentage
    - far_pct: Cumulative (exact + close + far) percentage
    - mismatch_pct: Mismatch percentage (non-cumulative)
    """
    total = len(df_subset)
    if total == 0:
        return {'exact_pct': 0, 'close_pct': 0, 'far_pct': 0, 'mismatch_pct': 0, 'total': 0}

    # Count each match type
    counts = df_subset['MatchType'].value_counts()
    exact = counts.get('exact', 0)
    close = counts.get('close', 0)
    far = counts.get('far', 0)
    mismatch = counts.get('mismatch', 0)
    no_pred = counts.get('no_prediction', 0)

    # Calculate percentages (excluding no_prediction from denominator for match rates)
    valid_total = total - no_pred
    if valid_total == 0:
        return {'exact_pct': 0, 'close_pct': 0, 'far_pct': 0, 'mismatch_pct': 0, 'total': total}

    exact_pct = exact / valid_total * 100
    close_pct = (exact + close) / valid_total * 100  # Cumulative
    far_pct = (exact + close + far) / valid_total * 100  # Cumulative
    mismatch_pct = mismatch / valid_total * 100

    return {
        'exact_pct': exact_pct,
        'close_pct': close_pct,
        'far_pct': far_pct,
        'mismatch_pct': mismatch_pct,
        'total': total,
        'valid_total': valid_total
    }


def evaluate_success_rates(df):
    """
    Calculate Success Rate (SR).
    SR = (Count of non-null predictions / Total count) * 100
    Evaluates both Food Code inference and Portion Size estimation.
    """
    print("\n" + "=" * 70)
    print("    Usability Evaluation: Success Rate (SR)")
    print("    (Proportion of images with ANY valid response)")
    print("=" * 70)

    # 1. Define what counts as a "Success"
    # For Food Codes: parsing the string returns a valid 8-digit code (not None)
    df['HasValidFoodCode'] = df['GPTFoodCode'].apply(lambda x: parse_gpt_food_code(x) is not None)

    # For Portion Sizes: string is not Na/None/Empty and not a no-answer phrase
    def is_valid_portion(val):
        if pd.isna(val) or str(val).strip() == '':
            return False
        return not any(p in str(val).strip().lower() for p in NO_PREDICTION_PHRASES)

    df['HasValidPortion'] = df['GPTPortionDescription'].apply(is_valid_portion)

    # 2. Prepare the Breakdown groups
    if 'PortionType' in df.columns:
        portion_types = df['PortionType'].dropna().unique()
        order = ['largest', 'medium', 'median', 'smallest']
        portion_types = sorted(portion_types, key=lambda x: order.index(x.lower()) if x.lower() in order else 99)
    else:
        portion_types = []

    # 3. Print Header
    print(f"{'Category':<15} {'Total Images':>12} {'Food Code':>15} {'Portion Size':>15}")
    print(f"{'':<15} {'':>12} {'SR (%)':>15} {'SR (%)':>15}")
    print("-" * 70)

    # 4. Calculate and Print Overall Stats
    total = len(df)
    fc_success = df['HasValidFoodCode'].sum()
    ps_success = df['HasValidPortion'].sum()

    fc_sr = (fc_success / total * 100) if total > 0 else 0
    ps_sr = (ps_success / total * 100) if total > 0 else 0

    print(f"{'OVERALL':<15} {total:>12} {fc_sr:>15.1f} {ps_sr:>15.1f}")

    # 5. Calculate and Print Breakdown by Portion Type
    if len(portion_types) > 0:
        print("-" * 70)
        for ptype in portion_types:
            subset = df[df['PortionType'] == ptype]
            sub_total = len(subset)

            if sub_total == 0:
                continue

            sub_fc = subset['HasValidFoodCode'].sum()
            sub_ps = subset['HasValidPortion'].sum()

            sub_fc_sr = (sub_fc / sub_total * 100)
            sub_ps_sr = (sub_ps / sub_total * 100)

            print(f"{ptype.capitalize():<15} {sub_total:>12} {sub_fc_sr:>15.1f} {sub_ps_sr:>15.1f}")

    print("=" * 70)


def print_portion_type_metrics(df_eval):
    """Print food code metrics breakdown by PortionType."""
    print("\n" + "=" * 70)
    print("    Food Code Performance by Portion Type (Cumulative Metrics)")
    print("=" * 70)
    print(f"{'Portion Type':<15} {'Exact':>12} {'Close':>12} {'Far':>12} {'Mismatch':>12}")
    print(f"{'':<15} {'match (%)':>12} {'match (%)':>12} {'match (%)':>12} {'(%)':>12}")
    print("-" * 70)

    # Get unique portion types and sort them
    portion_types = df_eval['PortionType'].dropna().unique()
    order = ['largest', 'medium', 'median', 'smallest']
    portion_types = sorted(portion_types, key=lambda x: order.index(x.lower()) if x.lower() in order else 99)

    for ptype in portion_types:
        subset = df_eval[df_eval['PortionType'] == ptype]
        metrics = compute_cumulative_metrics(subset)

        print(f"{ptype.capitalize():<15} {metrics['exact_pct']:>12.1f} {metrics['close_pct']:>12.1f} "
              f"{metrics['far_pct']:>12.1f} {metrics['mismatch_pct']:>12.1f}")

    print("=" * 70)


def _weight_metrics_for_subset(df_subset):
    """Compute MAE, MAPE, median AE, and no-prediction count for a subset."""
    valid = df_subset.dropna(subset=['AbsError'])
    no_pred = df_subset['GPTWeight'].isna().sum()
    if len(valid) == 0:
        return {'mae': None, 'mape': None, 'median_ae': None, 'no_pred': int(no_pred)}
    return {
        'mae': float(valid['AbsError'].mean()),
        'mape': float(valid['PctError'].mean()),
        'median_ae': float(valid['AbsError'].median()),
        'no_pred': int(no_pred),
    }


def print_weight_error_metrics(df_eval):
    """Print portion size weight estimation error by portion type."""
    print("\n" + "=" * 70)
    print("    Portion Size Weight Estimation Error")
    print("=" * 70)

    portion_types = df_eval['PortionType'].dropna().unique() if 'PortionType' in df_eval.columns else []
    order = ['largest', 'medium', 'median', 'smallest']
    portion_types = sorted(portion_types, key=lambda x: order.index(x.lower()) if x.lower() in order else 99)

    # Header
    header = f"{'Metric':<20}"
    for ptype in portion_types:
        header += f"{ptype.capitalize():>15}"
    header += f"{'Total':>15}"
    print(header)
    print("-" * 70)

    # Compute metrics per portion type and overall
    col_stats = []
    for ptype in portion_types:
        col_stats.append(_weight_metrics_for_subset(df_eval[df_eval['PortionType'] == ptype]))
    col_stats.append(_weight_metrics_for_subset(df_eval))  # Total

    def fmt(val, decimals=1):
        return f"{val:>15.{decimals}f}" if val is not None else f"{'N/A':>15}"

    def fmt_int(val):
        return f"{val:>15}" if val is not None else f"{'N/A':>15}"

    for metric_key, label in [('mae', 'MAE (g)'), ('mape', 'MAPE (%)'), ('median_ae', 'Median AE (g)')]:
        row = f"{label:<20}"
        for s in col_stats:
            row += fmt(s[metric_key])
        print(row)

    # No prediction count
    row = f"{'No prediction (n)':<20}"
    for s in col_stats:
        row += fmt_int(s['no_pred'])
    print(row)

    print("=" * 70)


def print_summary(metrics, weight_metrics=None):
    """Print evaluation summary in formatted table."""
    print("\n" + "=" * 55)
    print("    Food Code Inference Evaluation (ASA24)")
    print("=" * 55)

    total = metrics['total_samples']
    valid = metrics['valid_predictions']

    print(f"\nTotal samples: {total} (rows with valid FoodCode)")
    print(f"Valid predictions: {valid} ({valid/total*100:.1f}%)")

    print("\n" + "-" * 55)
    print(f"{'Match Type':<20} {'Count':>10} {'Percentage':>15}")
    print("-" * 55)

    # Order of match types for display
    match_order = ['exact', 'close', 'far', 'mismatch', 'no_prediction']
    match_labels = {
        'exact': 'Exact Match',
        'close': 'Close Match (2-digit)',
        'far': 'Far Match (1-digit)',
        'mismatch': 'Mismatch',
        'no_prediction': 'No Prediction'
    }

    for match_type in match_order:
        count = metrics['counts'].get(match_type, 0)
        pct = metrics['percentages'].get(match_type, 0)
        label = match_labels[match_type]
        print(f"{label:<20} {count:>10} {pct:>14.1f}%")

    print("-" * 55)
    print(f"{'Total':<20} {total:>10} {100.0:>14.1f}%")
    print("=" * 55)

    # Print food code metrics by portion type
    df_eval = metrics['df_with_classifications']
    if 'PortionType' in df_eval.columns:
        print_portion_type_metrics(df_eval)

    # Print portion size weight error metrics
    if weight_metrics is not None:
        print_weight_error_metrics(weight_metrics['df_with_weights'])


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate food code and portion size weight error on ASA24 dataset'
    )
    parser.add_argument(
        '--input', '-i',
        default='../data/ASA24_GPTFoodCodes_portion.csv',
        help='Path to input CSV file with predictions'
    )
    parser.add_argument(
        '--weights', '-w',
        default='../data/FoodWeights.csv',
        help='Path to FoodWeights.csv for portion weight lookup'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path to save detailed results CSV (optional)'
    )
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include rows without valid FoodCode in evaluation'
    )

    args = parser.parse_args()

    # Load data
    print(f"\nLoading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Total rows in file: {len(df)}")

    print(f"Loading food weights from: {args.weights}")
    df_weights = pd.read_csv(args.weights)

    # Run food code evaluation
    filter_valid = not args.include_all
    food_code_metrics = evaluate_food_codes(df, filter_valid=filter_valid)

    # Run portion size weight evaluation
    weight_metrics = None
    required_cols = {'LabelAmount', 'LabelUnit', 'GPTPortionDescription', 'GPTPortionAmount', 'FoodCode'}
    if required_cols.issubset(df.columns):
        weight_metrics = evaluate_portion_weight(df, df_weights)
        print(f"\nPortion weight evaluation: {weight_metrics['n_valid']} valid rows, "
              f"{weight_metrics['no_prediction']} no prediction, "
              f"{weight_metrics['no_gt']} missing GT weight")
    else:
        missing = required_cols - set(df.columns)
        print(f"\nSkipping weight evaluation — missing columns: {missing}")

    # Print summary
    print_summary(food_code_metrics, weight_metrics)
    evaluate_success_rates(df)

    # Save detailed results if requested
    if args.output:
        df_results = food_code_metrics['df_with_classifications']
        # Merge weight columns if available
        if weight_metrics is not None:
            weight_cols = ['GTWeight', 'GPTWeight', 'AbsError', 'PctError']
            df_results = df_results.merge(
                weight_metrics['df_with_weights'][['FileName'] + weight_cols],
                on='FileName', how='left'
            )
        output_cols = ['FileName', 'FoodCode', 'GPTFoodDescription', 'GPTFoodCode', 'MatchType',
                       'PortionType', 'Portion', 'GPTPortionDescription', 'GPTPortionAmount',
                       'GTWeight', 'GPTWeight', 'AbsError', 'PctError']
        output_cols = [c for c in output_cols if c in df_results.columns]
        df_results[output_cols].to_csv(args.output, index=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
