from flask import Flask, request, jsonify, send_file, render_template
import os
import time
from joblib import load
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load your saved model
model_path = r"D:/projects/DeepLearning PY Project/Model/DDD_model.h5"
DDD_model = tf.keras.models.load_model(model_path)


app = Flask(__name__)

# Set the directory for uploaded images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploaded_images')

# Ensure the directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # Render the main HTML file


@app.route('/upload', methods=['POST'])
def upload_image():
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Clear existing images in the upload folder
    for file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save the uploaded image
    original_filename = image_file.filename
    original_file_path = os.path.join(UPLOAD_FOLDER, original_filename)
    image_file.save(original_file_path)

    # Preprocess the image
    processed_image_path = os.path.join(UPLOAD_FOLDER, f"processed_{original_filename}")
    img = Image.open(original_file_path).convert("L")  # Convert to grayscale
    img = img.resize((64, 64))  # Resize for preprocessing
    img.save(processed_image_path)

    # Return URLs for images
    original_image_url = f"/images/{original_filename}"
    processed_image_url = f"/images/processed_{original_filename}"
    return jsonify({
        "original_image_url": original_image_url,
        "processed_image_url": processed_image_url
    }), 200




@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    # Preprocess the image
    input_image = preprocess_image_for_model(image_path)  # This should be the same as preprocessing step for prediction


    try:
        # Predict using the model
        predictions = DDD_model.predict(input_image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Map predicted class to label (ensure that your classes match)
        classes = {0: "open", 1: "closed"}
        predicted_label = classes[predicted_class]

        return jsonify({"predicted_class": predicted_label})
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500




def preprocess_image(image_path ,apply_histogram_equalization=False):
    """Preprocess image for prediction."""
    # Load the image and convert to grayscale
    image = plt.imread(image_path)
    resized_image = cv2.resize(image, (64, 64))
  # Resize to match the model input dimensions
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Resize to match the model input
    return gray_image

def preprocess_image_for_model(image_path ):
    """Preprocess image for prediction."""
    # Load the image and convert to grayscale
    image = plt.imread(image_path)
    resized_image = cv2.resize(image, (64, 64))
  # Resize to match the model input dimensions
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray_image = np.expand_dims(gray_image, axis=-1)  # Add channel dimension
    input_image = np.expand_dims(gray_image, axis=0)  # Add batch dimension 
    return input_image


@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from the uploaded_images folder."""
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')  # Adjust MIME type if needed
    else:
        return jsonify({"error": "Image not found"}), 404


@app.route('/Start')
def result():
    return render_template('Start.html')


@app.route('/download/<filename>')
def download_file(filename):
    """Download the specified file."""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


@app.route("/loading")
def delay():
    """Simulate a delay for a loading animation."""
    time.sleep(3)
    return jsonify({"message": "Loading finished"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
