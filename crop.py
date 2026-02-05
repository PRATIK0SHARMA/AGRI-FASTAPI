from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
from io import BytesIO
import io
import joblib
import tensorflow as tf
from fastapi.responses import JSONResponse
import os
import sys
import json

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# 1. CRITICAL - Force TensorFlow 2.x behavior
os.environ['TF_USE_LEGACY_KERAS'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Force re-import with clean state
if 'tensorflow' in sys.modules:
   del sys.modules['tensorflow']
if 'keras' in sys.modules:
   del sys.modules['keras']

import os

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR: {BASE_DIR}")

# Check what's in the crop folder
print("Files in crop folder:")
for file in os.listdir(BASE_DIR):
   print(f"  - {file}")

# Check if translations folder exists
translations_path = os.path.join(BASE_DIR, "translations")
print(f"\ntranslations folder exists: {os.path.exists(translations_path)}")
if os.path.exists(translations_path):
   print("Files in translations folder:")
   for file in os.listdir(translations_path):
       print(f"  - {file}")

# Check if models folder exists
models_path = os.path.join(BASE_DIR, "models")
print(f"\nmodels folder exists: {os.path.exists(models_path)}")


app = FastAPI()

# Use absolute paths
CROP_MODEL_PATH = os.path.join(BASE_DIR, "models","xgb_compressed.joblib")
CROP_SCALER_PATH = os.path.join(BASE_DIR, "models", "minmax_scaler.pkl")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Crop_Disease_Model (1).keras")
CROP_TRANSLATIONS_PATH = os.path.join(BASE_DIR, "translations", "crop_pred_translations.json")
DISEASE_TRANSLATIONS_PATH = os.path.join(BASE_DIR, "translations", "crop_disease_translations.json")

print(f"\nCROP_TRANSLATIONS_PATH: {CROP_TRANSLATIONS_PATH}")
print(f"File exists: {os.path.exists(CROP_TRANSLATIONS_PATH)}")

# Get current directory
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Define new paths
# CROP_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "crop", "xgb_crop_model.pkl")
# CROP_SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "crop", "minmax_scaler.pkl")
# DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "disease", "Crop_Disease_Model.keras")
# CROP_TRANSLATIONS_PATH = os.path.join(BASE_DIR, "..", "translations", "crop_pred_translations.json")
# DISEASE_TRANSLATIONS_PATH = os.path.join(BASE_DIR, "..", "translations", "crop_disease_translations.json")

with open(CROP_TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
    crop_translations = json.load(f)

with open(DISEASE_TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
    disease_translations = json.load(f)

# Supported languages
SUPPORTED_LANGUAGES = { "english": "en", "eng": "en", "en": "en",
   "hindi": "hi", "hin": "hi", "hi": "hi",
   "bengali": "bn", "ben": "bn", "bn": "bn", "bangla": "bn",
   "telugu": "te", "tel": "te", "te": "te",
   "marathi": "mr", "mar": "mr", "mr": "mr",
   "tamil": "ta", "tam": "ta", "ta": "ta",
   "kannada": "kn", "kan": "kn", "kn": "kn",
   "punjabi": "pa", "pun": "pa", "pa": "pa", "panjabi": "pa"}

# Load XGBoost crop model and scaler
# crop_model = joblib.load(CROP_MODEL_PATH)
# crop_scaler = joblib.load(CROP_SCALER_PATH)
# print("Crop model and scaler loaded successfully!")

# # Load TensorFlow disease model
# tf.keras.backend.clear_session()
# try:
#    disease_model = tf.keras.models.load_model(
#        DISEASE_MODEL_PATH,
#        custom_objects={'KerasLayer': tf.keras.layers.Layer}
#    )
#    print("Disease model loaded successfully!")
# except Exception as e:
#    print(f"Error loading disease model: {e}")
#    disease_model = None

crop_model = None
crop_scaler = None
disease_model = None

def get_crop_model():
    global crop_model, crop_scaler
    if crop_model is None:
        crop_model = joblib.load(CROP_MODEL_PATH)
        crop_scaler = joblib.load(CROP_SCALER_PATH)
    return crop_model, crop_scaler

def load_disease_model():
    global disease_model
    if disease_model is None:
        print("Loading disease model...")
        tf.keras.backend.clear_session()
        disease_model = tf.keras.models.load_model(
            DISEASE_MODEL_PATH,
            custom_objects={'KerasLayer': tf.keras.layers.Layer}
        )
        print("Disease model loaded!")
    return disease_model

# Disease model labels
disease_labels = ['Corn___Common_Rust',
'Corn___Gray_Leaf_Spot',
'Corn___Healthy',
'Corn___Northern_Leaf_Blight',
'Potato___Early_Blight',
'Potato___Healthy',
'Potato___Late_Blight',
'Rice___Brown_Spot',
'Rice___Healthy',
'Rice___Leaf_Blast',
'Rice___Neck_Blast',
'Sugarcane_Bacterial Blight',
'Sugarcane_Healthy',
'Sugarcane_Red Rot',
'Wheat___Brown_Rust',
'Wheat___Healthy',
'Wheat___Yellow_Rust']



crop_labels=['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']


@app.post("/predict-crop")
async def predict_crop(N: int, P: int, K: int, temperature: float, humidity: float, ph: float, rainfall: float,language: str = "english"):
   # Prepare raw input in the exact order used during training
   # Order: N, P, K, temperature, humidity, ph, rainfall
   model, scaler = get_crop_model()
   if language not in SUPPORTED_LANGUAGES:
       language = "english"
   lang_code=SUPPORTED_LANGUAGES[language]
   raw_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=np.float32)

   # Scale input using the same scaler used during training
   scaled_input = scaler.transform(raw_input)

   # Get predictions from scaled input
   probabilities = model.predict_proba(scaled_input)[0]

   # Get top predictions
   top1_index = np.argmax(probabilities)
   predicted_crop = crop_labels[top1_index]

   top3_indices = np.argsort(probabilities)[-3:][::-1]
   #top3_crops = [crop_labels[i] for i in top3_indices]
   # USE TRANSLATIONS HERE!
   predicted_crop_translated = crop_translations[str(top1_index)][lang_code]

   top_3 = [
           {"class": crop_translations[str(i)][lang_code], "confidence": float(probabilities[i])}
           for i in top3_indices
       ]

   #return {"predicted_crop": predicted_crop, "top3_crops": top3_crops}
   return {
           "predicted_crop": predicted_crop_translated,
           "confidence": round(float(probabilities[top1_index]), 4),
           "top_3_predictions": top_3,
           "language": language
           #"all_predictions": predictions[0].tolist()  # For debugging
       }

@app.get("/")
async def root():
   return {
       "message": "Crop & Disease Prediction API",
       "endpoints": {
           "crop": "/predict-crop - POST with N, P, K, temperature, humidity, ph, rainfall parameters",
           "disease": "/predict-disease - POST with image file"
       }
   }

@app.post("/predict-disease")
async def predict_disease(
   file: UploadFile = File(None),          # For website/file upload
   base64_image: str = None,               # For mobile camera
   language: str = "english"
):
   try:
       # Validate language
       if language not in SUPPORTED_LANGUAGES:
           language = "english"
       lang_code = SUPPORTED_LANGUAGES[language]

       model = load_disease_model()  # Add this line
       
       if model is None:  # Change from disease_model to model
           return {"error": "Disease model failed to load"}

       image_bytes = None

       # Check which input method is used
       if file and file.filename:
           # Method 1: File upload
           image_bytes = await file.read()
       elif base64_image:
           # Method 2: Base64 from camera
           import base64
           image_bytes = base64.b64decode(base64_image)
       else:
           return {"error": "No image provided. Send either 'file' or 'base64_image'"}

       # Process image
       image = Image.open(io.BytesIO(image_bytes))

       if image.mode != 'RGB':
           image = image.convert('RGB')

       image = image.resize((224, 224))
       img_array = np.array(image).astype('float32')
       img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
       img_array = np.expand_dims(img_array, axis=0)

       # Make prediction
       predictions = model.predict(img_array, verbose=0)

       predicted_idx = np.argmax(predictions[0])
       top_3_indices = np.argsort(predictions[0])[-3:][::-1]

       # Get translations
       predicted_disease_translated = disease_translations[str(predicted_idx)][lang_code]

       top_3 = [
           {
               "class": disease_translations[str(i)][lang_code],
               "confidence": float(predictions[0][i])
           }
           for i in top_3_indices
       ]

       return {
           "predicted_disease": predicted_disease_translated,
           "confidence": round(float(predictions[0][predicted_idx]), 4),
           "top_3_predictions": top_3,
           "language": language
           # Removed "source": source_type - not needed
       }

   except Exception as e:
       return {"error": f"Prediction failed: {str(e)}"}