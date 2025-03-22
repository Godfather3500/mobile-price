from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import joblib
import numpy as np
from inference_sdk import InferenceHTTPClient
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
from io import BytesIO
from PIL import Image

app = FastAPI()

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

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = ""  # Path to your Google Drive API credentials

def authenticate_google_drive():
    """Authenticate and return Google Drive service."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def upload_to_google_drive(file_path: str):
    """Upload a file to Google Drive and return the shareable link."""
    drive_service = authenticate_google_drive()
    file_metadata = {'name': os.path.basename(file_path)}
    media = MediaFileUpload(file_path, resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    
    # Make the file publicly accessible
    drive_service.permissions().create(
        fileId=file['id'],
        body={'role': 'reader', 'type': 'anyone'}
    ).execute()

    # Generate the shareable link
    file_id = file['id']
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def get_condition_from_roboflow(image_url: str):
    """Get the condition of the device from the Roboflow API."""
    try:
        result = client.run_workflow(
            workspace_name="rough-gxilk",
            workflow_id="custom-workflow-2",
            images={"image": image_url},
            use_cache=True
        )
        condition = result[0]['predictions']['predictions'][0]['class']
        return condition.capitalize()
    except Exception as e:
        print(f"Roboflow API error: {str(e)}")
        return "Unknown"  # Fallback value

def predict_price_all_features(os: str, screen_size: float, five_g: str, internal_memory: int, ram: int, battery: int, release_year: int, days_used: int, normalized_new_price: float, device_brand: str, image_url: str):
    """Predict the resale price of a mobile device."""
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
    condition = get_condition_from_roboflow(image_url)

    # Apply decision-making logic to adjust final price
    price_adjustment = {"Good": 1.0, "Fair": 0.85, "Worst": 0.6}
    final_price = initial_price * price_adjustment.get(condition)

    return initial_price*1000, condition, final_price*1000

@app.post('/predict')
async def predict(
    os: str = Form(...),
    screen_size: float = Form(...),
    five_g: str = Form(...),
    internal_memory: int = Form(...),
    ram: int = Form(...),
    battery: int = Form(...),
    release_year: int = Form(...),
    days_used: int = Form(...),
    normalized_new_price: float = Form(...),
    device_brand: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Save the uploaded file temporarily
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # Upload the image to Google Drive
        image_url = upload_to_google_drive(temp_filename)

        # Predict the price
        initial_price, condition, final_price = predict_price_all_features(
            os, screen_size, five_g, internal_memory, ram, battery,
            release_year, days_used, normalized_new_price, device_brand, image_url
        )
        
        return {
            "Condition": condition,
            "Predicted Price": round(initial_price, 2),
            "Final Adjusted Price": round(final_price, 2),
            "Image URL": image_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)