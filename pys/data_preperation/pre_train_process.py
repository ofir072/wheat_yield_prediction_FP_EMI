import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np


def replace_with_nan_based_on_rik(df):
    feature_sets = {
        'סוג מחזור (שלחין / פלחה חרבה)': ['סוג מחזור (שלחין / פלחה חרבה)-שלחין', 'סוג מחזור (שלחין / פלחה חרבה)-פלחה',
                                          'סוג מחזור (שלחין / פלחה חרבה)-ריק'],
        'סוג כרב': ['סוג כרב-יבש', 'סוג כרב-רטוב', 'סוג כרב-ריק'],
        'גידול קודם מחולק לקטגוריות': ['גידול קודם מחולק לקטגוריות-אחר', 'גידול קודם מחולק לקטגוריות-קטניות',
                                       'גידול קודם מחולק לקטגוריות-דגנים ותפוח אדמה',
                                       'גידול קודם מחולק לקטגוריות-כותנה', 'גידול קודם מחולק לקטגוריות-גזר וסלק',
                                       'גידול קודם מחולק לקטגוריות-ריק'],
        'ייעוד החלקה (גרעינים / שחת / תחמיץ)': ['ייעוד החלקה (גרעינים / שחת / תחמיץ)-שחת',
                                                'ייעוד החלקה (גרעינים / שחת / תחמיץ)-תחמיץ',
                                                'ייעוד החלקה (גרעינים / שחת / תחמיץ)-ריק']
    }
    rik_columns_to_drop = []
    for prime_feature, sub_features in feature_sets.items():
        rik_column = f'{prime_feature}-ריק'
        if rik_column in df.columns:
            actual_sub_features = [col for col in sub_features if col != rik_column]
            for sub_feature in actual_sub_features:
                if sub_feature in df.columns:
                    df.loc[df[rik_column] == 1, sub_feature] = np.nan
            rik_columns_to_drop.append(rik_column)
    df.drop(columns=rik_columns_to_drop, inplace=True)
    return df


def split_and_save_model(data, column_name, model_name):
    model_data = data[data[column_name] == 1]
    baal_data = model_data[model_data['השקייה (מושקה/בעל)-בעל'] == 1]
    moshke_data = model_data[model_data['השקייה (מושקה/בעל)-מושקה'] == 1]

    if not baal_data.empty:
        baal_train, baal_test = train_test_split(baal_data, test_size=0.2, random_state=42)
        baal_train.to_excel(os.path.join(output_dir_train, f'{model_name}_בעל_train.xlsx'), index=False)
        baal_test.to_excel(os.path.join(output_dir_test, f'{model_name}_בעל_test.xlsx'), index=False)
    else:
        print(f"No בעל data found for model {model_name}. Skipping.")

    if not moshke_data.empty:
        moshke_train, moshke_test = train_test_split(moshke_data, test_size=0.2, random_state=42)
        moshke_train.to_excel(os.path.join(output_dir_train, f'{model_name}_מושקה_train.xlsx'), index=False)
        moshke_test.to_excel(os.path.join(output_dir_test, f'{model_name}_מושקה_test.xlsx'), index=False)
    else:
        print(f"No משוקה data found for model {model_name}. Skipping.")


def split_and_save_chita(data, chita_column, yiaud_column):
    chita_data = data[data[chita_column] == 1]
    chita_grain_data = chita_data[chita_data[yiaud_column] == 1]
    chita_non_grain_data = chita_data[chita_data[yiaud_column] == 0]

    chita_grain_baal = chita_grain_data[chita_grain_data['השקייה (מושקה/בעל)-בעל'] == 1]
    chita_grain_moshke = chita_grain_data[chita_grain_data['השקייה (מושקה/בעל)-מושקה'] == 1]
    chita_non_grain_baal = chita_non_grain_data[chita_non_grain_data['השקייה (מושקה/בעל)-בעל'] == 1]
    chita_non_grain_moshke = chita_non_grain_data[chita_non_grain_data['השקייה (מושקה/בעל)-מושקה'] == 1]

    if not chita_grain_baal.empty:
        chita_grain_baal_train, chita_grain_baal_test = train_test_split(chita_grain_baal, test_size=0.2, random_state=42)
        chita_grain_baal_train.to_excel(os.path.join(output_dir_train, 'חיטה_גרעינים_בעל_train.xlsx'), index=False)
        chita_grain_baal_test.to_excel(os.path.join(output_dir_test, 'חיטה_גרעינים_בעל_test.xlsx'), index=False)
    else:
        print("No grain בעל data found for חיטה. Skipping.")

    if not chita_grain_moshke.empty:
        chita_grain_moshke_train, chita_grain_moshke_test = train_test_split(chita_grain_moshke, test_size=0.2, random_state=42)
        chita_grain_moshke_train.to_excel(os.path.join(output_dir_train, 'חיטה_גרעינים_מושקה_train.xlsx'), index=False)
        chita_grain_moshke_test.to_excel(os.path.join(output_dir_test, 'חיטה_גרעינים_מושקה_test.xlsx'), index=False)
    else:
        print("No grain מושקה data found for חיטה. Skipping.")

    if not chita_non_grain_baal.empty:
        chita_non_grain_baal_train, chita_non_grain_baal_test = train_test_split(chita_non_grain_baal, test_size=0.2, random_state=42)
        chita_non_grain_baal_train.to_excel(os.path.join(output_dir_train, 'חיטה_שחת_תחמיץ_בעל_train.xlsx'), index=False)
        chita_non_grain_baal_test.to_excel(os.path.join(output_dir_test, 'חיטה_שחת_תחמיץ_בעל_test.xlsx'), index=False)
    else:
        print("No non-grain בעל data found for חיטה. Skipping.")

    if not chita_non_grain_moshke.empty:
        chita_non_grain_moshke_train, chita_non_grain_moshke_test = train_test_split(chita_non_grain_moshke, test_size=0.2, random_state=42)
        chita_non_grain_moshke_train.to_excel(os.path.join(output_dir_train, 'חיטה_שחת_תחמיץ_מושקה_train.xlsx'), index=False)
        chita_non_grain_moshke_test.to_excel(os.path.join(output_dir_test, 'חיטה_שחת_תחמיץ_מושקה_test.xlsx'), index=False)
    else:
        print("No non-grain מושקה data found for חיטה. Skipping.")


def split_and_save_rel_chita(data, chita_column, yiaud_column):
    chita_data = data[data[chita_column] == 1]
    chita_grain_data = chita_data[chita_data[yiaud_column] == 1]

    chita_grain_data.loc['יבול-  (ק\"ג/ד\')'] = chita_grain_data['יבול-  (ק\"ג/ד\')'] / (chita_grain_data['יבול קש (ק\"ג/ד\')'] + chita_grain_data['יבול-  (ק\"ג/ד\')'])
    chita_grain_data = chita_grain_data[chita_grain_data['יבול-  (ק\"ג/ד\')'] != 1]
    chita_rel_grain_baal = chita_grain_data[chita_grain_data['השקייה (מושקה/בעל)-בעל'] == 1]
    chita_rel_grain_moshke = chita_grain_data[chita_grain_data['השקייה (מושקה/בעל)-מושקה'] == 1]

    columns_to_drop_rel_grain = ['יבול קש (ק\"ג/ד\')']

    chita_rel_grain_baal = chita_rel_grain_baal.drop(columns=columns_to_drop_rel_grain)
    chita_rel_grain_moshke = chita_rel_grain_moshke.drop(columns=columns_to_drop_rel_grain)

    if not chita_rel_grain_baal.empty:
        chita_rel_grain_baal_train, chita_rel_grain_baal_test = train_test_split(chita_rel_grain_baal, test_size=0.2, random_state=42)
        chita_rel_grain_baal_train.to_excel(os.path.join(output_dir_train, 'חיטה_יחס_גרעינים_בעל_train.xlsx'), index=False)
        chita_rel_grain_baal_test.to_excel(os.path.join(output_dir_test, 'חיטה_יחס_גרעינים_בעל_test.xlsx'), index=False)
    else:
        print("No grain בעל data found for חיטה. Skipping.")

    if not chita_rel_grain_moshke.empty:
        chita_rel_grain_moshke_train, chita_rel_grain_moshke_test = train_test_split(chita_rel_grain_moshke, test_size=0.2, random_state=42)
        chita_rel_grain_moshke_train.to_excel(os.path.join(output_dir_train, 'חיטה_יחס_גרעינים_מושקה_train.xlsx'), index=False)
        chita_rel_grain_moshke_test.to_excel(os.path.join(output_dir_test, 'חיטה_יחס_גרעינים_מושקה_test.xlsx'), index=False)
    else:
        print("No grain מושקה data found for חיטה. Skipping.")


def split_and_save_model_top_grain(data, column_name, model_name):
    if "שחת" in model_name:
        model_data = data[data[column_name] != 1]
    else:
        model_data = data[data[column_name] == 1]

    baal_data = model_data[model_data['השקייה (מושקה/בעל)-בעל'] == 1]
    moshke_data = model_data[model_data['השקייה (מושקה/בעל)-מושקה'] == 1]

    if not baal_data.empty:
        baal_train, baal_test = train_test_split(baal_data, test_size=0.2, random_state=42)
        baal_train.to_excel(os.path.join(output_dir_train, f'{model_name}_אחוזון_בעל_train.xlsx'), index=False)
        baal_test.to_excel(os.path.join(output_dir_test, f'{model_name}_אחוזון_בעל_test.xlsx'), index=False)
    else:
        print(f"No בעל data found for model {model_name}. Skipping.")

    if not moshke_data.empty:
        moshke_train, moshke_test = train_test_split(moshke_data, test_size=0.2, random_state=42)
        moshke_train.to_excel(os.path.join(output_dir_train, f'{model_name}_אחוזון_מושקה_train.xlsx'), index=False)
        moshke_test.to_excel(os.path.join(output_dir_test, f'{model_name}_אחוזון_מושקה_test.xlsx'), index=False)
    else:
        print(f"No משוקה data found for model {model_name}. Skipping.")


if __name__ == "__main__":
    # Load the dataset
    location_file = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\FE\ims_fe.xlsx'
    df = pd.read_excel(location_file)

    # Define the output directories
    output_dir_test = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\test'
    output_dir_train = r'C:\Users\אופיר גוטליב\PycharmProjects\wheat_yield_prdiction_FP_EMI\data\models_sets\train'

    # Define the binary columns for each model
    models = {
        'שעורה': 'גידול נוכחי-שעורה',
        'אפונה': 'גידול נוכחי-אפונה',
        'תלתן': 'גידול נוכחי-תלתן'
    }
    chita_columns = 'גידול נוכחי-חיטה'
    yiaud_columns = 'ייעוד החלקה (גרעינים / שחת / תחמיץ)-גרעינים'

    top_grain_models = {
        'שעורה': 'גידול נוכחי-שעורה',
        'חיטה_גרעינים': 'ייעוד החלקה (גרעינים / שחת / תחמיץ)-גרעינים',
        'חיטה_שחת_תחמיץ': 'ייעוד החלקה (גרעינים / שחת / תחמיץ)-גרעינים'
    }

    # Iterate over each model and split the data
    for models_name, columns_name in models.items():
        split_and_save_model(df, columns_name, models_name)

    # Additional splits for חיטה
    split_and_save_chita(df, chita_columns, yiaud_columns)

    # Additional splits for חיטה ליחס גרעינים
    split_and_save_rel_chita(df, chita_columns, yiaud_columns)

    # Iterate over each model and split the data to top grain models
    for models_name, columns_name in top_grain_models.items():
        split_and_save_model_top_grain(df, columns_name, models_name)
