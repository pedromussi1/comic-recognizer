import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

model = load_model('comic_book_classifier_model.h5')

# Define class labels (adjust this to your specific labels)
class_labels = ['Avengers', 'Batman Dark Victory', 'Batman Hush', "Batman The Long Halloween", 'Batman Year One', 'Fantastic Four 1', 'Fantastic Four 2', 'House of X', 'New X-Men', 'Nightwing 1', 'Nightwing 2', 'Nightwing 3', 'Nightwing 4', 'Watchmen', 'X-Men Fate of the Phoenix', 'X-Men Proteus', 'X-Men Second Genesis']  # Example labels

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Resize the image to the target size
    img = cv2.resize(img, (224, 224))  # Assuming your model expects 224x224 input size
    
    # Normalize the image to be between 0 and 1
    img = img.astype('float32') / 255.0
    
    # Expand dimensions to match the input shape of the model (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            print(f"File saved to {file_path}")
            img = preprocess_image(file_path)
            prediction = model.predict(img)
            predicted_class = class_labels[np.argmax(prediction)]
            print(f"Predicted class: {predicted_class}")
            return render_template('result.html', label=predicted_class)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
