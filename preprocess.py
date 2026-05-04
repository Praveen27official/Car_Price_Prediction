import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()

    # Drop Car_Name
    df = df.drop(['Car_Name'], axis=1)

    # Create new feature
    df['Car_Age'] = 2024 - df['Year']
    df = df.drop(['Year'], axis=1)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df
