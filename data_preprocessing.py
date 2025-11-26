import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_and_select_columns(df, selected_columns):
    df_selected = df[selected_columns].copy()
    return df_selected

def encode_abilities(df):
    df = df.copy()
    df["abilities"] = df["abilities"].apply(ast.literal_eval)
    mlb = MultiLabelBinarizer()
    abilities_encoded = mlb.fit_transform(df["abilities"])
    abilities_df = pd.DataFrame(abilities_encoded, columns=mlb.classes_)
    df = pd.concat([df, abilities_df], axis=1)
    df.drop(columns=["abilities"], inplace=True)
    return df


