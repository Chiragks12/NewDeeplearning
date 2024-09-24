from flask import Flask, request, render_template
import numpy as np
import os
import cv2
from keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Config
model_file = "models/image_classification_model.h5"
# model_file = "models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "images"
dic = {0 : 'Cat', 1 : 'Dog'}

# Load model
model = load_model(model_file)
model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(150,150))
	i = img_to_array(i)/255.0
	i = i.reshape(1, 150,150,3)
	prediction_prob = model.predict(i)[0][0]
	if prediction_prob < 0.5: # 0-0.5: cat, else dog
		output = "cat"
	else:
		output = "dog"

	return output

@app.route('/', methods=['GET', 'POST'])
def hello():
	return render_template('index.html')

@app.route('/submit', methods=[ 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "images/" + img.filename	
		img.save(img_path)
		output = predict_label(img_path)
	return render_template("index.html", prediction = output, img_path = img_path)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port = 8080)