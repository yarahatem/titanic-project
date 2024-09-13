from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(df):
    X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Example usage:
# df = preprocess_data(load_data('data/train.csv'))
# model, accuracy = train_model(df)
# print(f"Model accuracy: {accuracy}")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def compare_models(X_train, y_train, X_test, y_test):
    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name} accuracy: {accuracy}")

# Example usage:
# compare_models(X_train, y_train, X_test, y_test)
