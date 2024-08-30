import numpy as np
import cv2
import tensorflow as tf
import pickle

def load_model_and_encoder(model_path, encoder_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the label encoder
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

def preprocess_image(image_path, img_size):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_class(model, img, label_encoder):
    # Predict class probabilities
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

def main():
    model_path = 'comic_book_classifier_model.h5'
    encoder_path = 'label_encoder.pkl'
    img_path = 'path_to_your_comic_cover_image.jpg'  # Replace with your image path
    
    # Load model and label encoder
    model, label_encoder = load_model_and_encoder(model_path, encoder_path)
    
    # Preprocess the image
    img_size = 224  # Should match the size used in preprocessing
    img = preprocess_image(img_path, img_size)
    
    # Predict and print the result
    predicted_label = predict_class(model, img, label_encoder)
    print(f'Predicted Comic Book: {predicted_label}')

if __name__ == "__main__":
    main()
