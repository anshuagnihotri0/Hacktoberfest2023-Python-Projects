import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model

# Load the pre-trained SRGAN model
srgan_model = load_model('path_to_pretrained_SRGAN_model.h5')

# Load and preprocess the low-resolution image
input_image_path = 'path_to_low_resolution_image.jpg'
input_image = cv2.imread(input_image_path)
input_image = cv2.resize(input_image, (32, 32))  # Resize image to match the input shape of the SRGAN model
input_image = preprocess_input(input_image)

# Perform image resolution enhancement using the pre-trained SRGAN model
upscaled_image = srgan_model.predict(tf.expand_dims(input_image, axis=0))[0]

# Display the original and upscaled images
cv2.imshow('Low Resolution Image', input_image)
cv2.imshow('Enhanced Image', upscaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
