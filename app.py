from flask import Flask,  request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the saved model
model = tf.keras.models.load_model('./model/2')

# Class labels
class_labels = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Rice_bacterial_leaf_blight',
 'Rice_brown_spot',
 'Rice_healthy',
 'Rice_leaf_blast',
 'Rice_leaf_scald',
 'Rice_narrow_brown_spot']

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    resized_image = image.resize((256, 256))
    # Convert the image to a numpy array
    img_array = np.array(resized_image)
    # Reshape the image array to match the model's expected input shape

    img_array = img_array.reshape(1,256,256,3)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is an image
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read the image file
        img = Image.open(file.stream)
        # Preprocess the image
        img_array = preprocess_image(img)
        # Perform prediction
        predictions = model.predict(img_array)
        # Get predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)
        # Get predicted class name from class labels
        predicted_class_name = class_labels[predicted_class_index[0]]
        # Return the predicted class name along with the prediction
        return jsonify({'prediction': predicted_class_name})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
