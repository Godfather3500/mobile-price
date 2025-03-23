from flask import Flask, request, jsonify
import joblib
import os
import tempfile
import numpy as np
from inference_sdk import InferenceHTTPClient
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model and scalers
model = joblib.load("mobile_model.pkl")
feature_scaler = joblib.load("scaler.pkl")
scaler = joblib.load("scaler_y.pkl")
label_encoder_os = joblib.load("label_encoder_os.pkl")
label_encoder_5g = joblib.load("label_encoder_5g.pkl")
brand_averages = joblib.load("brand_averages.pkl")

# Initialize Roboflow Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Function to send the image to Roboflow API
def get_condition_from_roboflow(image_path):
    """ Send image to Roboflow and get condition """
    with open(image_path, "rb") as img_file:   # Open in binary mode
        result = client.run_workflow(
            workspace_name="rough-gxilk",
            workflow_id="custom-workflow-2",
            images={"image": img_file},         # Send file object
            use_cache=True
        )

    # Extract condition from predictions
    condition = result[0]['predictions']['predictions'][0]['class']
    return condition.capitalize()

# Function to predict price and condition
def predict_price_all_features(os, screen_size, five_g, internal_memory, ram, battery, release_year, days_used, normalized_new_price, device_brand, image_path):
    """ Predict mobile price and condition """
    
    brand_average = brand_averages.get(device_brand)

    def group_brand_pred(value):
        if value > 4.7:
            return 0
        elif 4.3 < value <= 4.7:
            return 1
        elif 4.0 < value <= 4.3:
            return 2
        elif 3.7 < value <= 4.0:
            return 3
        else:
            return 4

    brand_group = group_brand_pred(brand_average)

    os_encoded = label_encoder_os.transform([os])[0]
    five_g_encoded = label_encoder_5g.transform([five_g])[0]

    input_data = np.array([[
        os_encoded,
        screen_size,
        five_g_encoded,
        internal_memory,
        ram,
        battery,
        release_year,
        days_used,
        normalized_new_price,
        brand_group
    ]])

    # Scale input features
    scaled_input = feature_scaler.transform(input_data)
    prediction_scaled = model.predict(scaled_input)
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    initial_price = prediction[0][0]

    # Get condition from Roboflow
    condition = get_condition_from_roboflow(image_path)

    # Apply price adjustment
    price_adjustment = {"Good": 1.0, "Fair": 0.85, "Worst": 0.6}
    final_price = initial_price * price_adjustment.get(condition, 1.0)

    return initial_price * 1000, condition, final_price * 1000

@app.route('/predict', methods=['POST'])
def predict():
    """ Flask route to handle predictions """
    try:
        # Handle file upload
        if 'image' not in request.files:
            return jsonify({"error": "No image part"}), 400

        image = request.files['image']

        if image.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            image_path = temp_file.name
            image.save(image_path)

        # Extract other form data
        data = request.form
        initial_price, condition, final_price = predict_price_all_features(
            data["os"], float(data["screen_size"]), data["five_g"],
            float(data["internal_memory"]), float(data["ram"]), float(data["battery"]),
            int(data["release_year"]), int(data["days_used"]),
            float(data["normalized_new_price"]), data["device_brand"], image_path
        )

        # Remove the temp image after processing
        os.remove(image_path)

        return jsonify({
            "Condition": condition,
            "Predicted Price": round(initial_price, 2),
            "Final Adjusted Price": round(final_price, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
