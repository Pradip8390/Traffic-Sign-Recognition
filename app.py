from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("model.h5")

# Define class labels
class_labels = ["Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing","No passing for vechiles over 3.5 metric tons",
                "Right-of-way at the next intersection","Priority road","Yield","Stop","No vechiles","Vechiles over 3.5 metric tons prohibited","No entry","General caution","Dangerous curve to the left","Dangerous curve to the right",
                "Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow","Wild animals crossing","End of all speed and passing limits",
                "Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory","End of no passing","End of no passing by vechiles over 3.5 metric tons"]  

# Preprocessing function for images9
def preprocess_image(image):
    # Resize the image to the same dimensions as the training data
    image = cv2.resize(image, (32, 32))
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Equalize the histogram
    image = cv2.equalizeHist(image)
    # Normalize the image
    image = image / 255.0
    # Reshape to match model input
    image = image.reshape(1, 32, 32, 1)
    return image

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting traffic signal class
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', prediction="No file uploaded", confidence="N/A")

    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', prediction="No file selected", confidence="N/A")

    # Read the uploaded image
    image = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return render_template('result.html', prediction=class_labels[predicted_class], confidence=f"{confidence * 100:.2f}%")

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
