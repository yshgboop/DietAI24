import argparse

import numpy as np
import pandas as pd


DEFAULT_NUTRIENTS = [
    'Energy (kcal)',
    'Protein (g)',
    'Carbohydrate (g)',
    'Total Fat (g)',
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Estimate nutrient totals from portion-with-weights CSV.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../data/ASA24_GPTFoodCodes_portion_with_weights.csv',
        help='Input CSV with CalculatedWeight and CalculatedWeightGPT columns',
    )
    parser.add_argument(
        '--nutrition',
        type=str,
        default='../FNDDS/2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx',
        help='FNDDS nutrient Excel workbook',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/ASA24_GPTFoodCodes_nutrition.csv',
        help='Output CSV path',
    )
    return parser.parse_args()


def get_nutrition_values(food_code, weight, df_nutrition, col_names_nutrition):
    nutrition_values = {}
    if pd.isna(weight) or weight < 0:
        return nutrition_values

    food_code_nutrition = df_nutrition[df_nutrition['Food code'] == food_code]
    if len(food_code_nutrition) == 1:
        for nutrition_name in col_names_nutrition:
            nutrition_values[nutrition_name] = (
                food_code_nutrition[nutrition_name].values[0] * weight * 0.01
            )
    else:
        print(f"Food code {food_code} not found in nutrition data")
    return nutrition_values


def main():
    args = parse_arguments()

    df_results = pd.read_csv(args.input)
    df_nutrition = pd.read_excel(
        args.nutrition,
        sheet_name='FNDDS Nutrient Values',
        skiprows=1,
    )
    col_names_nutrition = [
        col for col in df_nutrition.columns[4:] if col in DEFAULT_NUTRIENTS
    ]

    nutrient_cols = {}
    for nutrient_name in col_names_nutrition:
        nutrient_cols[nutrient_name] = np.nan
        nutrient_cols[f'{nutrient_name}_GPT'] = np.nan

    df_results = pd.concat(
        [df_results, pd.DataFrame(nutrient_cols, index=df_results.index)],
        axis=1,
    )

    for i in range(len(df_results)):
        food_code = df_results.loc[i, 'FoodCode']

        weight_label = df_results.loc[i, 'CalculatedWeight']
        nutrition_values_label = get_nutrition_values(
            food_code, weight_label, df_nutrition, col_names_nutrition
        )
        for nutrient_name, value in nutrition_values_label.items():
            df_results.loc[i, nutrient_name] = value

        weight_gpt = df_results.loc[i, 'CalculatedWeightGPT']
        nutrition_values_gpt = get_nutrition_values(
            food_code, weight_gpt, df_nutrition, col_names_nutrition
        )
        for nutrient_name, value in nutrition_values_gpt.items():
            df_results.loc[i, f'{nutrient_name}_GPT'] = value

    df_results.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")
    print(f"Total rows processed: {len(df_results)}")


if __name__ == '__main__':
    main()
