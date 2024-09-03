import tensorflow as tf
import os
import shutil
from PIL import Image
import numpy as np

# Paths
model_path = '/Users/amandanassar/Desktop/V60 NOK AI/my_model.h5'
data_dir = '/Users/amandanassar/Desktop/V60 NOK AI/data/Data images'
output_dirs = {
    'camerafault': '/Users/amandanassar/Desktop/V60 NOK AI/data/camerafault',
    'realNOK': '/Users/amandanassar/Desktop/V60 NOK AI/data/realNOK',
    'external': '/Users/amandanassar/Desktop/V60 NOK AI/data/external'
}

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to preprocess images
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize((224, 224))  # Convert to RGB and resize
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        print(f"Preprocessed image shape: {img_array.shape}")
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# Function to classify and move image
def classify_and_move_image(image_path):
    try:
        img_array = preprocess_image(image_path)
        if img_array is None:
            return

        prediction = model.predict(img_array)
        print(f"Prediction for {image_path}: {prediction}")

        class_index = np.argmax(prediction[0])
        print(f"Predicted class index: {class_index}")

        # Reverse class mapping
        # Original mapping: ['camerafault', 'realNOK', 'external']
        # Desired mapping: ['camerafault', 'external', 'realNOK']
        reversed_class_names = ['camerafault', 'external', 'realNOK']

        # Ensure class_index is valid
        if class_index >= len(reversed_class_names):
            print(f"Class index {class_index} is out of bounds.")
            return

        class_name = reversed_class_names[class_index]
        print(f"Class name: {class_name}")

        # Create directory if it doesn't exist
        if not os.path.exists(output_dirs[class_name]):
            os.makedirs(output_dirs[class_name])

        # Move the file
        shutil.move(image_path, os.path.join(output_dirs[class_name], os.path.basename(image_path)))
        print(f"Moved {image_path} to {class_name}.")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process all images
for file_name in os.listdir(data_dir):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        classify_and_move_image(os.path.join(data_dir, file_name))
