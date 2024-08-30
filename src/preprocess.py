import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import replace_background

def preprocess_images(input_dir, output_dir, augment=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                
                # Optional: Replace background
                new_background = np.random.randint(0, 255, img.shape, dtype=np.uint8)
                img = replace_background(img, new_background)
                
                # Resize and save
                img_resized = cv2.resize(img, (224, 224))  # Match input size for transfer learning
                output_path = os.path.join(output_dir, folder)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                
                # Save the original resized image
                cv2.imwrite(os.path.join(output_path, img_file), img_resized)
                
                # Perform augmentation
                if augment:
                    img_array = np.expand_dims(img_resized, axis=0)
                    i = 0
                    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_path, save_prefix='aug', save_format='jpg'):
                        i += 1
                        if i > 5:  # Generate 5 augmented images per original image
                            break

if __name__ == "__main__":
    preprocess_images("dataset", "preprocessed_dataset")
