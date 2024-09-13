import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Fill missing values and encode categorical data
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

# Example usage:
# df = load_data('data/train.csv')
# df = preprocess_data(df)
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/train.csv')

# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.show()

# Correlation matrix
sns.heatmap(df.corr(), annot=True)
plt.show()
