from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow React to communicate with Flask

# Load the machine learning model (ensure this file exists in the same directory)
try:
    model = joblib.load("priority_model_with_categories.pkl")  # Make sure the model file is available
except FileNotFoundError:
    print("Model file not found. Ensure 'priority_model_with_categories.pkl' is in the same directory.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check if the required fields are present
    if 'text' not in data or 'description' not in data or 'category' not in data:
        return jsonify({"error": "Missing fields: 'text', 'description', and/or 'category'"}), 400

    # Extract the task details from the received JSON
    text = data['text']
    description = data['description']
    category = data['category']

    # Prepare the input data for the model (convert into a DataFrame)
    input_data = pd.DataFrame([[text, description, category]], columns=['text', 'description', 'category'])

    try:
        # Make the prediction (model should be pre-trained)
        predicted_priority = model.predict(input_data)[0]  # Get the predicted priority from the model
        return jsonify({"predicted_priority": predicted_priority})  # Return the prediction as a JSON response
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
