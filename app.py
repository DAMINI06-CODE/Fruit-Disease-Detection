from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
try:
    model = load_model('fruit-disease-detection/apple_disease_detection_model.keras')  # Replace with actual path
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image could not be read")
        img = cv2.resize(img, (224, 224))  # Assuming input shape for the model
        img = img / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Expand dimensions for batch
        return img
    except Exception as e:
        logger.error(f"Error occurred while preprocessing image: {e}")
        return None

# Function to predict disease based on uploaded image
def predict_disease(image_path):
    try:
        img = preprocess_image(image_path)
        if img is None:
            return None
        prediction = model.predict(img)
        logger.info(f"Prediction shape: {prediction.shape}")
        logger.info(f"Prediction data: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Error occurred while predicting disease: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({'error': 'No file part in the request'})
        
        f = request.files['image']
        if f.filename == '':
            logger.warning("No image selected for uploading")
            return jsonify({'error': 'No image selected for uploading'})
        
        filename = secure_filename(f.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(image_path)

        prediction = predict_disease(image_path)
        if prediction is None:
            logger.error("Failed to predict disease")
            return jsonify({'error': 'Failed to predict disease'})
        
        # Update class names to match model's classes
        class_names = [
            'Alternaria', 'Anthracnose', 'blackspot','Black Mould Rot', 'Blotch_Apple', 'cankar', 
            'fresh', 'greening', 'Healthy', 'Normal_Apple', 'Rot_Apple', 
            'Scab_Apple', 'Stem end Rot'
        ]
        if len(prediction) == 0 or len(prediction[0]) == 0:
            raise ValueError("Prediction result is empty")

        predicted_class_index = np.argmax(prediction[0])
        if predicted_class_index >= len(class_names):
            raise ValueError(f"Prediction index {predicted_class_index} out of bounds for class names")

        predicted_class = class_names[predicted_class_index]
        
        response = {
            'filename': f"/static/uploads/{filename}",
            'predicted_class': predicted_class
        }
        return render_template('result.html', result=response)
    except Exception as e:
        logger.error(f"Error occurred during prediction request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
