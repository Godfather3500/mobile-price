from flask import Flask, request, jsonify
import joblib
import numpy as np
from inference_sdk import InferenceHTTPClient
import os
from werkzeug.utils import secure_filename
import tempfile
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

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
    api_key="KKuCYpsib9sFBf5SKENX"
)

# Function to get condition from Roboflow API
def get_condition_from_roboflow(image_path):
    """Send image to Roboflow and get predicted condition label."""
    result = client.run_workflow(
            workspace_name="batman-4caqy",
            workflow_id="custom-workflow",
            images={
                "image": image_path
            },
            use_cache=True  # Cache workflow definition for 15 minutes
        )

    print("📸 Roboflow response:", result)  # Optional: for debugging

        # Extract the predicted class label
    predicted_classes = result[0]['predictions'].get('predicted_classes', [])
    condition = predicted_classes[0]
    return condition.capitalize()


# Prediction function
def predict_price_all_features(os, screen_size, five_g, internal_memory, ram, battery, release_year, days_used, normalized_new_price, device_brand, image_path):
    """Predict price and condition from image."""
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

    scaled_input = feature_scaler.transform(input_data)
    prediction_scaled = model.predict(scaled_input)
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    initial_price = prediction[0][0]

    # Get condition from Roboflow API
    condition = get_condition_from_roboflow(image_path)

    # Apply decision-making logic to adjust final price
    price_adjustment = {"Good": 1.0, "Fair": 0.85, "Worst": 0.6}
    final_price = initial_price * price_adjustment.get(condition)

    return initial_price * 1000, condition, final_price * 1000

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form data
        os = request.form.get("os")
        screen_size = float(request.form.get("screen_size"))
        five_g = request.form.get("five_g")
        internal_memory = int(request.form.get("internal_memory"))
        ram = int(request.form.get("ram"))
        battery = int(request.form.get("battery"))
        release_year = int(request.form.get("release_year"))
        days_used = int(request.form.get("days_used"))
        normalized_new_price = float(request.form.get("normalized_new_price"))
        device_brand = request.form.get("device_brand")

        # Handle image upload
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image = request.files["image"]

        if image.filename == "":
            return jsonify({"error": "No image selected"}), 400

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            image_path = tmp_file.name

        # Make prediction
        initial_price, condition, final_price = predict_price_all_features(
            os, screen_size, five_g, internal_memory, ram, battery,
            release_year, days_used, normalized_new_price, device_brand, image_path
        )

        # Clean up: Remove the temporary image
       # os.remove(image_path)

        # Return response
        return jsonify({
            "Condition": condition,
            "Predicted Price": round(initial_price, 2),
            "Final Adjusted Price": round(final_price, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
