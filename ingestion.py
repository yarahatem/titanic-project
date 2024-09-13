import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

