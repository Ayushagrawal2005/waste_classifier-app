from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer as BaseInputLayer
import numpy as np
import os
import secrets
import logging
import sys
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting application initialization...")
logger.info(f"TensorFlow version: {tf.__version__}")

@register_keras_serializable('Custom')
class CustomInputLayer(BaseInputLayer):
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['input_shape'] = config.pop('batch_shape')[1:]
        return cls(**config)

@register_keras_serializable('Custom')
class CustomAdam(Adam):
    def __init__(self, *args, weight_decay=None, **kwargs):
        if weight_decay is not None:
            kwargs['decay'] = weight_decay
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        if 'weight_decay' in config:
            config['decay'] = config.pop('weight_decay')
        return config

# Initialize model as None
model = None
class_labels = ['plastic', 'green-glass', 'other']

def create_custom_objects():
    custom_objects = {
        'InputLayer': CustomInputLayer,
        'DTypePolicy': Policy,
        'float32': tf.float32,
        'Adam': CustomAdam
    }
    return custom_objects

def load_model_with_custom_objects():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "waste_classifier_model.h5")
        logger.info(f"Attempting to load model from: {model_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Model file exists: {os.path.exists(model_path)}")
        
        custom_objects = create_custom_objects()
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            loaded_model = load_model(model_path, compile=False)  # Load without compiling first
            logger.info("Model architecture loaded")
            
            # Recompile with compatible optimizer
            loaded_model.compile(
                optimizer=CustomAdam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("Model compiled with custom optimizer")
            return loaded_model
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.exception("Full stack trace:")
        raise

try:
    model = load_model_with_custom_objects()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    sys.exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Required for sessions
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///waste_classifier.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    points = db.Column(db.Integer, default=0)
    uploads = db.relationship('ImageUpload', backref='user', lazy=True)

class ImageUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    is_verified = db.Column(db.Boolean, default=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

# Health check endpoint
@app.route("/api/health", methods=["GET"])
def health_check():
    if model is None:
        return jsonify({"status": "unhealthy", "message": "Model not loaded"}), 503
    return jsonify({
        "status": "healthy",
        "message": "API is running",
        "model_loaded": model is not None
    }), 200

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    error = None
    points_added = 0
    
    # Ensure user exists in session
    if 'user_id' not in session:
        # Create new user
        new_user = User()
        db.session.add(new_user)
        db.session.commit()
        session['user_id'] = new_user.id

    user = User.query.get(session['user_id'])

    if request.method == "POST":
        if 'image' not in request.files:
            error = "No file uploaded"
        else:
            file = request.files["image"]
            if file.filename == '':
                error = "No file selected"
            else:
                try:
                    # Create unique filename
                    filename = f"upload_{secrets.token_hex(8)}_{file.filename}"
                    filepath = os.path.join("static", filename)
                    file.save(filepath)
                    image_url = filepath
                    logger.debug(f"File saved to: {filepath}")

                    # Load and preprocess image
                    img = image.load_img(filepath, target_size=(224, 224))
                    img_array = image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    logger.debug("Image loaded and preprocessed successfully")

                    # Predict
                    result = model.predict(img_array)
                    predicted_class = class_labels[np.argmax(result)]
                    confidence = float(np.max(result))
                    
                    # Add points if confidence is high (verification)
                    if confidence > 0.85:  # High confidence threshold
                        points_to_add = 10
                        user.points += points_to_add
                        points_added = points_to_add
                    
                    # Save upload record
                    upload = ImageUpload(
                        filename=filename,
                        prediction=predicted_class,
                        confidence=confidence,
                        is_verified=confidence > 0.85,
                        user_id=user.id
                    )
                    db.session.add(upload)
                    db.session.commit()

                    prediction = f"Predicted class: {predicted_class} (Confidence: {confidence:.2%})"
                    logger.info(f"Prediction made: {prediction}")

                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    logger.error(f"Error during prediction: {str(e)}")
                    logger.exception("Full stack trace:")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    image_url = None

    return render_template(
        "index.html", 
        prediction=prediction, 
        error=error, 
        image_url=image_url,
        user_points=user.points,
        points_added=points_added
    )

if __name__ == "__main__":
    if model is None:
        logger.error("Cannot start server: Model failed to load")
        sys.exit(1)
        
    logger.info("Starting waste classification API server...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)  # Set debug=False for production
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)
