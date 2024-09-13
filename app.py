from flask import Flask, request, jsonify
from model import train_model
from ingestion import load_data, preprocess_data

app = Flask(__name__)

# Load and preprocess data
df = preprocess_data(load_data('data/train.csv'))
model, _ = train_model(df)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    X_new = pd.DataFrame([data])
    prediction = model.predict(X_new)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
