from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import secrets
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = load_model("waste_clssifier.py/waste_classifier_model.h5")
class_labels = ['plastic', 'green-glass', 'other']

# Generate a secure API key - this should be stored securely in production
API_KEY = secrets.token_urlsafe(32)
logger.info(f"Your API Key: {API_KEY}")  # This will be logged when server starts

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_api_key():
    api_key = request.headers.get('X-API-Key')
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt from {request.remote_addr}")
        return False
    return True

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    image_url = None
    if request.method == "POST":
        if 'image' not in request.files:
            error = "No file uploaded"
        else:
            file = request.files["image"]
            if file.filename == '':
                error = "No file selected"
            elif not allowed_file(file.filename):
                error = "Invalid file type. Please upload a PNG or JPEG image."
            else:
                try:
                    # Create static folder if it doesn't exist
                    os.makedirs('static', exist_ok=True)
                    
                    filepath = os.path.join("static", file.filename)
                    file.save(filepath)
                    image_url = filepath  # Store the image path

                    # Load and preprocess image
                    img = image.load_img(filepath, target_size=(224, 224))
                    img_array = image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predict
                    result = model.predict(img_array)
                    predicted_class = class_labels[np.argmax(result)]
                    prediction = f"Predicted class: {predicted_class}"

                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    image_url = None

    return render_template("index.html", prediction=prediction, error=error, image_url=image_url)

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route("/api/predict", methods=["POST"])
def predict_api():
    client_ip = request.remote_addr
    logger.info(f"Received prediction request from {client_ip}")

    if not verify_api_key():
        return jsonify({"error": "Invalid or missing API key"}), 401

    if 'image' not in request.files:
        logger.warning(f"No file uploaded in request from {client_ip}")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type attempted from {client_ip}")
        return jsonify({"error": "Invalid file type. Please upload a PNG or JPEG image"}), 400

    try:
        os.makedirs('static', exist_ok=True)
        filepath = os.path.join("static", f"temp_{secrets.token_hex(8)}_{file.filename}")
        file.save(filepath)
        logger.info(f"Successfully saved uploaded file to {filepath}")

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        predicted_class = class_labels[np.argmax(result)]
        confidence = float(np.max(result))

        response_data = {
            "prediction": predicted_class,
            "confidence": confidence,
            "class_probabilities": {
                label: float(prob) 
                for label, prob in zip(class_labels, result[0])
            }
        }

        logger.info(f"Successful prediction for {client_ip}: {predicted_class}")

        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing request from {client_ip}: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("Starting waste classification API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)