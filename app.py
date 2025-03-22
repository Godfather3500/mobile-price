from flask import Flask, request, jsonify
import joblib
import numpy as np
from inference_sdk import InferenceHTTPClient


# Initialize Flask app
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
    api_key="ugoLHHO11vI5X3Z4MvDI"
)

def get_condition_from_roboflow(image_path):
    result = client.run_workflow(
        workspace_name="rough-gxilk",
        workflow_id="custom-workflow-2",
        images={"image": image_path},
        use_cache=True
    )
    condition = result[0]['predictions']['predictions'][0]['class']
    return condition.capitalize()

def predict_price_all_features(os, screen_size, five_g, internal_memory, ram, battery, release_year, days_used, normalized_new_price, device_brand, image_path):
    
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

    return initial_price*1000, condition, final_price*1000

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_path = data.get("image_path")
        
        initial_price, condition, final_price = predict_price_all_features(
            data["os"], data["screen_size"], data["five_g"],
            data["internal_memory"], data["ram"], data["battery"],
            data["release_year"], data["days_used"], data["normalized_new_price"], data["device_brand"], image_path
        )
        
        return jsonify({
            "Condition": condition,
            "Predicted Price": round(initial_price, 2),
            "Final Adjusted Price": round(final_price, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
