from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('Blood-2')

# Define a route for receiving prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    input_data = np.array(data['input'])

    # Perform prediction using the loaded model
    predictions = model.predict(input_data)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
