# General import
!pip install -Uq grpcio==1.26.0
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import requests
import json
import tempfile

# TensorFlow Imports
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense, Dropout
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar_10

class_names =["airplane","automobile","bird","cat","deer","dog",
			"frog","horse", "ship","truck"]
# load and preprocessdataset
def load_and_preprocess():
   (x_train, y_train), (x_test,y_test) = cifar_10.load_data()
   y_train = to_categorical(y_train)
   y_test = to_categorical(y_test)
   x_train = x_train.astype('float32')
   x_test = x_test.astype('float32')
   x_train = x_train/255
   x_test = x_test/255
   return (x_train, y_train), (x_test,y_test)

# define model architecture
def get_model():
	model = Sequential([
	Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
	Conv2D(32, (3, 3), activation='relu', padding='same'),
	MaxPooling2D((2, 2)),
	Dropout(0.2),
	Conv2D(64, (3, 3), activation='relu', padding='same'),
	Conv2D(64, (3, 3), activation='relu', padding='same'),
	MaxPooling2D((2, 2)),
	Dropout(0.2),
	Flatten(),
	Dense(64, activation='relu'),
	Dense(10, activation='softmax')
	])

	model.compile(
	optimizer=SGD(learning_rate= 0.01 , momentum=0.1), 
	loss='categorical_crossentropy',
	metrics=['accuracy']
	)
	model.summary()
	return model
# train model
model = get_model()
model.fit(x_train, y_train,epochs=100,validation_data=(x_test, y_test))

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))
 
save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True
)