import os
import numpy as np
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers


def resample_folder(input_folder, output_folder, target_size=(256, 256)):
    """
    Resample images in the input folder and save them to the output folder.

    Parameters:
    - input_folder (str): Path to the input folder containing images.
    - output_folder (str): Path to the output folder to save resampled images.
    - target_size (tuple): Target size for resampling, e.g., (width, height).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of files in the input folder
    file_list = os.listdir(input_folder)

    # Process each file and save the resampled version
    for file_name in tqdm(file_list, desc="Resampling", unit="file"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Open the image
        img = Image.open(input_path)

        # Resize the image to the desired dimensions
        resampled_img = img.resize(target_size, Image.ANTIALIAS)

        # Save the resampled image
        resampled_img.save(output_path)

    print(f"Resampling completed for {input_folder}.")
    
# Function to load and preprocess the dataset
def load_data(sat_folder, gt_folder, num_classes=5, batch_size=1):
    input_images = []
    output_masks = []

    filenames = [filename for filename in os.listdir(sat_folder) if filename.endswith(".tif")]
    num_batches = len(filenames) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Loading batches", unit="batch"):
        batch_filenames = filenames[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        
        batch_inputs = []
        batch_outputs = []
        for filename in batch_filenames:
            input_path = os.path.join(sat_folder, filename)
            output_path = os.path.join(gt_folder, filename.replace("sat_", "gt_"))

            # Load and preprocess input image
            input_image = np.array(Image.open(input_path)) / 255.0  # Normalize to [0, 1]
            batch_inputs.append(input_image)

            # Load and preprocess output mask
            output_mask = np.array(Image.open(output_path))

            # Ensure that class indices are within the range [0, num_classes-1]
            output_mask = np.clip(output_mask, 0, num_classes - 1)
            output_mask = keras.utils.to_categorical(output_mask, num_classes=num_classes)
            batch_outputs.append(output_mask)

        input_images.append(np.array(batch_inputs))
        output_masks.append(np.array(batch_outputs))

    return np.vstack(input_images), np.vstack(output_masks)

# Define simple model for 5 classes
def simple_model(input_size=(256, 256, 3), num_classes=5):
    inputs = keras.Input(input_size)
    
    # Encoder
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Decoder
    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv2)
    up1 = layers.UpSampling2D((2, 2))(conv2)

    # Output layer with softmax activation for multi-class classification
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(up1)

    model = keras.Model(inputs, outputs)
    return model