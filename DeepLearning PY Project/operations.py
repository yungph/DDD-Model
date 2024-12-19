# import tensorflow as tf
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # Define the class labels dictionary
# classes = {
#     0: "open",
#     1: "closed",
# }

# # Path to the image you want to predict
# image_path = 'D:/projects/DeepLearning PY Project/uploaded_images/340.jpg'

# # Read the image using matplotlib
# image = plt.imread(image_path)

# # Resize the image to (64, 64) using OpenCV
# resized_image = cv2.resize(image, (64, 64))

# # Check the model's input shape


# model_path = r"D:\projects\DeepLearning PY Project\Model\DDD_model.h5"
# DDD_model = tf.keras.models.load_model(model_path)


# print(f"Model input shape: {DDD_model.input_shape}")

# # Preprocess the image according to the model's expected input
# # if DDD_model.input_shape[-1] == 3:  # If the model expects RGB
# #     input_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
# # DDD_model.input_shape[-1] == 1:  # If the model expects grayscale
# gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
# gray_image = np.expand_dims(gray_image, axis=-1)  # Add channel dimension
# input_image = np.expand_dims(gray_image, axis=0)  # Add batch dimension


# # Validate input shape
# print(f"Input shape for prediction: {input_image.shape}")

# # Predict the class of the image
# predictions = DDD_model.predict(input_image)

# # Convert predicted probabilities to class index
# predicted_class = np.argmax(predictions, axis=1)[0]

# # Retrieve the class label from the dictionary
# predicted_label = classes[predicted_class]

# print(f"Actual Class: closed")
# print(f"Predicted Class: {predicted_label}")
