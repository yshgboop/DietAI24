import pandas as pd
import numpy as np
import re

def parse_food_items_with_portions(food_string, food_codes_string, portion_weights_df):
    """
    Parses FNDDS portion format and looks up weights from FoodWeights.csv.

    Input format:
        GPTAmount: "chicken: 1.5 cup\nrice: 2 tablespoon"
        GPTFoodCode: "chicken: 27446400; rice: 20081000"

    Returns:
        Dictionary: {ingredient_name: weight_in_grams}
    """
    if pd.isna(food_string) or food_string == "I can't help to analyze this text.":
        return np.nan

    if pd.isna(food_codes_string):
        return np.nan

    # Parse food codes: "ingredient: code; ingredient2: code2" -> {ingredient: code}
    food_codes_dict = {}
    for item in food_codes_string.split(';'):
        if ':' in item:
            parts = item.split(':', 1)
            ingredient = parts[0].strip().lower()
            code = parts[1].strip()
            if code and code != "unknown":
                try:
                    food_codes_dict[ingredient] = int(code)
                except ValueError:
                    continue

    # Parse portion amounts: "ingredient: amount unit" -> {ingredient: (amount, unit)}
    food_dict = {}
    lines = food_string.split('\n')

    for line in lines:
        if not line.strip():
            continue

        try:
            if ':' not in line:
                continue

            name, portion_info = line.split(':', 1)
            name = name.strip().lower()
            portion_info = portion_info.strip()

            # Handle "unknown" portions
            if 'unknown' in portion_info.lower():
                food_dict[name] = np.nan
                continue

            # Parse "amount unit" format (e.g., "1.5 cup", "2 tablespoon", "6 leaf")
            parts = portion_info.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Could not parse portion: {portion_info}")
                food_dict[name] = np.nan
                continue

            amount_str, unit = parts
            try:
                amount = float(amount_str)
            except ValueError:
                print(f"Could not parse amount: {amount_str}")
                food_dict[name] = np.nan
                continue

            # Look up portion weight from FoodWeights.csv
            if name not in food_codes_dict:
                print(f"No food code found for ingredient: {name}")
                food_dict[name] = np.nan
                continue

            food_code = food_codes_dict[name]

            # Match food code and portion unit in FoodWeights.csv
            # First try exact match with the unit as-is
            matched = portion_weights_df[
                (portion_weights_df['FoodCode'] == food_code) &
                (portion_weights_df['Portion'].str.contains(unit, case=False, na=False, regex=False))
            ]

            if matched.empty:
                # Try matching with "1 " prefix (e.g., unit="cup" -> match "1 cup")
                matched = portion_weights_df[
                    (portion_weights_df['FoodCode'] == food_code) &
                    (portion_weights_df['Portion'] == f"1 {unit}")
                ]

            if matched.empty:
                print(f"No portion weight found for {name} (code: {food_code}, unit: {unit})")
                food_dict[name] = np.nan
                continue

            # Get the portion weight per unit (in grams)
            portion_weight = matched['Portion weight (g)'].iloc[0]

            # Calculate total weight
            total_weight = amount * portion_weight
            food_dict[name] = total_weight

        except Exception as e:
            print(f"Error processing line: {line}, {e}")
            continue

    return food_dict


def parse_food_items(food_string):
    """
    Legacy function for backward compatibility with gram-based format.
    Parses a string containing food items and their weights in grams.
    """
    food_string = food_string.split('\n\n')[0]
    if food_string == "I can't help to analyze this text.":
        return np.nan
    lines = food_string.split('\n')

    food_dict = {}
    for line in lines:
        try:
            name, weight = line.split(':')
            match = re.search(r'\d+', weight)
            if 'none' in weight.lower():
                food_dict[name.strip()] = np.nan
            elif match:
                food_dict[name.strip()] = float(match.group(0))
            else:
                print(f"Error in parsing weight: {weight}")
        except ValueError as e:
            print(f"Error processing line: {line}, {e}")

    return food_dict

def match_food_codes_na(food_string):
    """
    Matches food items to None weights.
    Handles new format: "ingredient: code; ingredient2: code2"
    """
    food_codes = {}
    items = food_string.split('; ')
    for item in items:
        try:
            if ':' not in item:
                continue
            name, code = item.split(':', 1)
            name = name.strip().lower()
            code = code.strip()
            # Store as list for compatibility with downstream code
            if code and code != "unknown":
                food_codes[name] = [code]
        except ValueError as e:
            print(f"Error processing item: {item}, {e}")

    food_weights = {food: np.nan for food in food_codes.keys()}
    return food_weights

def match_food_codes(food_string, weight_dict):
    """
    Matches food items to their respective weights.

    Handles new format: "ingredient: code; ingredient2: code2"
    Each ingredient maps to a single food code.
    """
    normalized_weights = {key.lower(): value for key, value in weight_dict.items()}

    food_codes = {}
    items = food_string.split('; ')
    for item in items:
        try:
            if ':' not in item:
                continue
            name, code = item.split(':', 1)
            name = name.strip().lower()
            code = code.strip()
            # Store as list for compatibility with downstream code
            if code and code != "unknown":
                food_codes[name] = [code]
        except ValueError as e:
            print(f"Error processing item: {item}, {e}")

    food_weights = {food: (codes, normalized_weights.get(food, np.nan)) for food, codes in food_codes.items()}
    return food_weights

def calculate_dish_nutrition(food_weights, nutrition_df, col_names_nutrition):
    """
    Calculates the total nutrition for a dish based on food weights and nutrition values.
    """
    nutrition_totals = pd.DataFrame(columns=col_names_nutrition)
    food_code_error = 0
    weight_error = 0
    weight_total = 0

    for food, (codes, weight) in food_weights.items():
        try:
            codes_float = [float(code) for code in codes if code.isdigit()]
            food_nutrition = nutrition_df[nutrition_df['Food code'].isin(codes_float)][col_names_nutrition]
            
            if pd.isna(weight):
                weight_error += 1
                print(f"Weight is None for {food}")
                continue
            
            weight_total += weight
            if not food_nutrition.empty:
                mean_nutrition = food_nutrition.mean()
                nutrition_for_food = mean_nutrition * (weight / 100)
                nutrition_totals = nutrition_totals.add(pd.DataFrame([nutrition_for_food]), fill_value=0)
        except ValueError as e:
            print(f"Error processing food item: {food}, {e}")
            food_code_error += 1
    
    return nutrition_totals, food_code_error, weight_error, weight_total

def process_results(df_results, df_nutrition, col_names_nutrition, output_path='../dish_metadata.csv'):
    """
    Processes the results DataFrame to calculate nutrition values for each dish.
    """
    ls_food_code_err = []
    ls_weight_err = []
    ls_ingredient_num = []
    ls_weight_est = []

    for i, row in df_results.iterrows():
        food_string = row['GPTFoodCode']
        weight_dict = row['GPTAmountWeight']

        if pd.isna(weight_dict):
            food_weights = match_food_codes_na(food_string)
            ls_food_code_err.append(0)
            ls_weight_err.append(len(food_weights))
            ls_ingredient_num.append(len(food_weights))
            ls_weight_est.append(np.nan)
        else:
            food_weights = match_food_codes(food_string, weight_dict)
            nutrition_dish, food_code_error_dish, weight_error_dish, weight_dish = calculate_dish_nutrition(food_weights, df_nutrition, col_names_nutrition)
            ls_food_code_err.append(food_code_error_dish)
            ls_weight_err.append(weight_error_dish)
            ls_ingredient_num.append(len(food_weights))
            ls_weight_est.append(weight_dish)

            if not nutrition_dish.empty:
                df_results.loc[i, col_names_nutrition] = nutrition_dish.values[0]

    df_results['FoodCodeError'] = ls_food_code_err
    df_results['WeightError'] = ls_weight_err
    df_results['IngredientNum'] = ls_ingredient_num
    df_results['GPTWeight'] = ls_weight_est

    df_results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate nutrition values from FNDDS portion estimates")
    parser.add_argument("--results_file", required=True,
                        help="Path to results CSV with GPTFoodCode and GPTAmount columns")
    parser.add_argument("--food_weights_file", required=True,
                        help="Path to FoodWeights.csv with FNDDS portion weights")
    parser.add_argument("--nutrition_file", required=True,
                        help="Path to FNDDS nutrient values Excel file")
    parser.add_argument("--output_file", required=True,
                        help="Path to save output CSV with nutrition values")
    parser.add_argument("--nutrients", nargs='+',
                        default=['Energy (kcal)', 'Protein (g)', 'Carbohydrate (g)', 'Total Fat (g)'],
                        help="List of nutrients to calculate (default: Energy, Protein, Carbs, Fat)")

    args = parser.parse_args()

    print("="*70)
    print("Nutrition Estimation from FNDDS Portions")
    print("="*70)

    # Load data
    print(f"\nLoading results from: {args.results_file}")
    df_results = pd.read_csv(args.results_file)
    print(f"Loaded {len(df_results)} dishes")

    print(f"\nLoading FoodWeights from: {args.food_weights_file}")
    portion_weights_df = pd.read_csv(args.food_weights_file)
    print(f"Loaded {len(portion_weights_df)} portion weight records")

    print(f"\nLoading FNDDS nutrition data from: {args.nutrition_file}")
    df_nutrition = pd.read_excel(args.nutrition_file, sheet_name='FNDDS Nutrient Values', skiprows=1)
    print(f"Loaded nutrition data for {len(df_nutrition)} food codes")

    # Filter to requested nutrients only
    available_nutrients = [col for col in df_nutrition.columns[4:] if col in args.nutrients]
    print(f"\nCalculating nutrients: {', '.join(available_nutrients)}")

    # Parse food items with FNDDS portions
    print("\nParsing portion sizes and looking up weights...")
    df_results['GPTAmountWeight'] = df_results.apply(
        lambda row: parse_food_items_with_portions(
            row['GPTAmount'],
            row['GPTFoodCode'],
            portion_weights_df
        ),
        axis=1
    )

    # Initialize columns for nutrition values
    nutrient_cols = {}
    for nutrient_name in available_nutrients:
        nutrient_cols[nutrient_name] = np.nan

    # Add all columns at once using pd.concat to avoid fragmentation
    df_results = pd.concat([df_results, pd.DataFrame(nutrient_cols, index=df_results.index)], axis=1)

    # Process results
    print("\nCalculating nutrition values for each dish...")
    process_results(df_results, df_nutrition, available_nutrients, output_path=args.output_file)

    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Total dishes processed: {len(df_results)}")
    print(f"  Nutrients calculated: {', '.join(available_nutrients)}")
    print(f"  Results saved to: {args.output_file}")
