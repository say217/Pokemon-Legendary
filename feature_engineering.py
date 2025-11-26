import pandas as pd

def encode_classification(df):
    class_dummies = pd.get_dummies(df["classfication"], prefix="class")
    df = pd.concat([df, class_dummies], axis=1)
    df.drop(columns=["classfication"], inplace=True)
    return df

def encode_type1(df):
    type1_dummies = pd.get_dummies(df["type1"], prefix="type1")
    df = pd.concat([df, type1_dummies], axis=1)
    df.drop(columns=["type1"], inplace=True)
    return df


