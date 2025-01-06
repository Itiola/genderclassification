from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model("model/gender_classifier_efficientnetb0.keras")

# Define class names
class_names = ["Female", "Male"]

# Upload folder path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the upload folder exists, if not, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the image
    img = Image.open(filepath)
    img = img.resize((224, 224))  # Resize to model input size

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Perform prediction
    predictions = model.predict(img_array)
    confidence = predictions[0][0]
    
    # Determine gender class
    if confidence >= 0.5:
        result = class_names[1]  # Male
        confidence_score = confidence * 100
    else:
        result = class_names[0]  # Female
        confidence_score = (1 - confidence) * 100

    return render_template('index.html', 
                           filename=file.filename, 
                           prediction=result, 
                           confidence=round(confidence_score, 2))

if __name__ == '__main__':
    app.run(debug=True)
