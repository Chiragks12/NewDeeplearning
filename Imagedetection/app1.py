from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import os
import cv2
import logging
from tensorflow.keras.models import load_model # type: ignore
from flask import send_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Initialize Flask app
app = Flask(__name__)

# Configure folder to save images
app.config['UPLOAD_FOLDER'] = 'images'

# Ensure the images folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Dictionary for labels
dic = {0: 'Cat', 1: 'Dog'}

# Load the model once when the app starts
try:
    model = load_model('models/image_classification_model.h5')
    model.make_predict_function()  # Important for TensorFlow/Keras predict() in Flask
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Set up logging to capture errors
logging.basicConfig(filename='logs/app.log', level=logging.DEBUG)


# Function to predict image label
def predict_label(img_path):
   def predict_label(img_path):
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(128, 128))  # Resize to 128x128
        img = img_to_array(img) / 255.0
        img = img.reshape(1, 128, 128, 3)  # Reshape for the model input

        # Perform the prediction
        prediction_prob = model.predict(img)[0][0]

        # Map prediction to label
        if prediction_prob < 0.5:  # 0-0.5: cat, else dog
            return "Cat"
        else:
            return "Dog"
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Error during prediction"


# Home route
@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template('index.html')

@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to handle file upload and prediction
@app.route('/submit', methods=['POST'])
def get_output():
    try:
        if 'my_image' not in request.files:
            return redirect(request.url)

        img = request.files['my_image']

        # Ensure the image is present
        if img.filename == '':
            return redirect(request.url)

        # Save the image to the specified folder
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        # Predict the label
        output = predict_label(img_path)

        # Render the result on the HTML template
        return render_template('index.html', prediction=output, img_path=img_path)

    except Exception as e:
        logging.error(f"Error in submission: {e}")
        return "Internal Server Error", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
