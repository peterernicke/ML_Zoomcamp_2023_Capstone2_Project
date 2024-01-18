import os
import random
import rasterio
import numpy as np
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

# Randomly sample ground truth files
def random_sample_files(folder, num_samples):
    all_files = os.listdir(folder)
    return random.sample(all_files, min(num_samples, len(all_files)))

# Visualize class distribution from a subset of ground truth files
def visualize_class_distribution(folder, num_samples=100, title='Class Distribution'):
    # Mapping from class labels to class names
    class_names = {
        0: 'unlabeled',
        1: 'buildings',
        2: 'woodlands',
        3: 'water',
        4: 'road'
    }

    # Mapping from class names to custom colors
    class_colors = {
        'unlabeled': 'gray',
        'buildings': 'red',
        'woodlands': 'forestgreen',
        'water': 'blue',
        'road': 'brown'
    }

    sampled_files = random_sample_files(folder, num_samples)
    class_co_occurrence_matrix = np.zeros((len(class_names), len(class_names)))

    class_counts = {}

    for file in tqdm(sampled_files, desc="Processing files", unit="file"):
        file_path = os.path.join(folder, file)

        with rasterio.open(file_path) as dataset:
            image = dataset.read(1)
        
        flattened_image = image.flatten()
        unique_classes, class_label_counts = zip(*[(label, list(flattened_image).count(label)) for label in set(flattened_image)])

        for class_label, count in zip(unique_classes, class_label_counts):
            # Convert class label to class name
            class_name = class_names.get(class_label, f'Class_{class_label}')
            
            if class_name not in class_counts:
                class_counts[class_name] = count
            else:
                class_counts[class_name] += count
        
        if len(unique_classes) > 1:
            # Update the co-occurrence matrix for each pair of classes
            for i in range(len(unique_classes)):
                for j in range(i + 1, len(unique_classes)):
                    class_i = class_names.get(unique_classes[i], f'Class_{unique_classes[i]}')
                    class_j = class_names.get(unique_classes[j], f'Class_{unique_classes[j]}')

                    class_co_occurrence_matrix[i, j] += class_label_counts[i]
                    class_co_occurrence_matrix[j, i] += class_label_counts[j]

    total_samples = sum(class_counts.values())
    class_frequencies = [count / total_samples for count in class_counts.values()]

    # Plot the class distribution bar chart with custom colors
    colors = [class_colors.get(class_name, 'gray') for class_name in class_counts.keys()]
    plt.bar(class_counts.keys(), class_frequencies, color=colors)
    plt.xlabel('Class Label')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

    # Pie chart with custom colors
    pie_colors = [class_colors.get(class_name, 'gray') for class_name in class_counts.keys()]
    plt.pie(class_frequencies, labels=class_counts.keys(), autopct='%1.1f%%', colors=pie_colors)
    plt.title('Class Distribution Pie Chart')
    plt.show()

    # Class distribution summary
    print("Class Distribution Summary:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} samples, {count/total_samples:.2%}")
    
    # Create a heatmap for class relationships with a logarithmic scale
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.log1p(class_co_occurrence_matrix), annot=True, cmap='Blues', xticklabels=class_names.values(), yticklabels=class_names.values())
    plt.xlabel('Class Label')
    plt.ylabel('Class Label')
    plt.title('Co-Occurrence of Classes')
    plt.show()

def visualize_image_pairs(sat_folder, gt_folder, num_samples=5):
    # Define colors for each class
    class_colors = {
        0: (0, 0, 0),        # unlabeled
        1: (255, 0, 0),      # buildings
        2: (34, 139, 34),    # woodlands
        3: (0, 0, 255),      # water
        4: (184, 115, 51)    # road
    }

    # Get a list of image files in the folders
    sat_files = os.listdir(sat_folder)
    gt_files = os.listdir(gt_folder)

    # Randomly sample image files
    sampled_files = random.sample(sat_files, min(num_samples, len(sat_files)))

    # Plot pairs of images
    for file in sampled_files:
        sat_image_path = os.path.join(sat_folder, file)
        gt_image_path = os.path.join(gt_folder, file.replace('sat_', 'gt_'))

        # Load images
        sat_image = Image.open(sat_image_path)
        gt_image = Image.open(gt_image_path)

        # Convert ground truth image to numpy array
        gt_array = np.array(gt_image)

        # Apply color mapping
        gt_colored = np.zeros(gt_array.shape + (3,), dtype=np.uint8)
        for class_label, color in class_colors.items():
            gt_colored[gt_array == class_label] = np.array(color)

        # Create a subplot with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the aerial image on the left
        axes[0].imshow(sat_image)
        axes[0].set_title('Aerial Image')
        axes[0].axis('off')

        # Plot the ground truth image on the right
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth Image')
        axes[1].axis('off')

        plt.show()
       
def resample_images(input_folder, output_folder, target_size=(256, 256)):
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
        #Image.ANTIALIAS: Anti-aliasing to reduce artifacts
        #Image.NEAREST: Nearest-neighbor sampling
        #Image.BOX: Box sampling
        #Image.BILINEAR: Bilinear interpolation
        #Image.HAMMING: Hamming-windowed sinc interpolation
        #Image.BICUBIC: Bicubic interpolation
        #Image.LANCZOS: Lanczos-windowed sinc interpolation
        resampled_img = img.resize(target_size, Image.ANTIALIAS)

        # Save the resampled image
        resampled_img.save(output_path)

    print("Resampling completed.")        

def visualize_images(sat_folder, gt_folder, model_path=None, num_samples=5):
    # Define colors for each class
    class_colors = {
        0: (0, 0, 0),        # unlabeled
        1: (255, 0, 0),      # buildings
        2: (34, 139, 34),    # woodlands
        3: (0, 0, 255),      # water
        4: (184, 115, 51)    # road
    }

    # Get a list of image files in the folders
    sat_files = os.listdir(sat_folder)
    gt_files = os.listdir(gt_folder)

    # Randomly sample image files
    sampled_files = random.sample(sat_files, min(num_samples, len(sat_files)))

    # Plot pairs or triples of images
    for file in sampled_files:
        sat_image_path = os.path.join(sat_folder, file)
        gt_image_path = os.path.join(gt_folder, file.replace('sat_', 'gt_'))

        # Load images
        sat_image = Image.open(sat_image_path)
        gt_image = Image.open(gt_image_path)

        # Convert ground truth image to numpy array
        gt_array = np.array(gt_image)

        # Apply color mapping for ground truth
        gt_colored = np.zeros(gt_array.shape + (3,), dtype=np.uint8)
        for class_label, color in class_colors.items():
            gt_colored[gt_array == class_label] = np.array(color)

        # Create a subplot with 1 or 2 columns based on the presence of the model
        if model_path:
            # Load the model
            model = keras.models.load_model(model_path)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Predict using the model
            input_image = np.array(sat_image) / 255.0
            input_example = np.expand_dims(input_image, axis=0)
            predictions = model.predict(input_example)
            predicted_mask = np.argmax(predictions[0], axis=-1)

            # Apply color mapping for prediction
            predicted_colored = np.zeros_like(gt_colored)
            for class_label, color in class_colors.items():
                predicted_colored[predicted_mask == class_label] = np.array(color)

            # Plot the predicted image on the right
            axes[2].imshow(predicted_colored)
            axes[2].set_title('Predicted Image')
            axes[2].axis('off')

        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the aerial image on the left
        axes[0].imshow(sat_image)
        axes[0].set_title('Aerial Image')
        axes[0].axis('off')

        # Plot the ground truth image in the middle
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth Image')
        axes[1].axis('off')

        plt.show()        
        
# Function to load and preprocess a batch of the dataset
def load_batch(sat_folder, gt_folder, filenames, target_size=(256, 256), num_classes=5):
    input_images = []
    output_masks = []

    for filename in filenames:
        sat_filename = filename
        gt_filename = filename.replace("sat_", "gt_")

        input_path = os.path.join(sat_folder, sat_filename)
        output_path = os.path.join(gt_folder, gt_filename)

        # Load and preprocess input image
        input_image = np.array(Image.open(input_path).resize(target_size)) / 255.0  # Normalize to [0, 1]
        input_images.append(input_image)

        # Load and preprocess output mask
        output_mask = np.array(Image.open(output_path).resize(target_size))
        
        # Ensure that class indices start from 0
        output_mask -= 1

        # Exclude class labels that are not present in the training set
        output_mask[output_mask >= num_classes] = 0

        output_mask = keras.utils.to_categorical(output_mask, num_classes=num_classes)
        output_masks.append(output_mask)

    return np.array(input_images), np.array(output_masks)

# Function to evaluate the model on a folder of images
def evaluate_model_on_folder(model, sat_folder, gt_folder, target_size=(256, 256)):
    # Get a list of image files in the folders
    sat_files = os.listdir(sat_folder)
    gt_files = os.listdir(gt_folder)

    # Ensure the order of files is the same
    sat_files.sort()
    gt_files.sort()

    # Load and preprocess the validation dataset
    X_val, y_val = load_batch(sat_folder, gt_folder, sat_files, target_size=target_size)

    # Predict on validation data
    y_pred = model.predict(X_val)

    # Convert predictions and ground truth to class labels
    y_pred_labels = np.argmax(y_pred, axis=-1)
    y_val_labels = np.argmax(y_val, axis=-1)

    # Flatten the 2D arrays to 1D
    y_pred_flat = y_pred_labels.flatten()
    y_val_flat = y_val_labels.flatten()

    # Define class names
    class_names = ["Unlabeled", "Buildings", "Woodlands", "Water", "Road"]

    # Compute confusion matrix
    conf_mat = confusion_matrix(y_val_flat, y_pred_flat)

    # Display confusion matrix as a heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_val_flat, y_pred_flat, target_names=class_names))        
    
# Function to load and preprocess a single example
def preprocess_example(sat_path):
    input_image = np.array(Image.open(sat_path)) / 255.0
    return np.expand_dims(input_image, axis=0)

# Function to visualize predictions for a given example
def visualize_example(sat_path, gt_path, model, num_classes):
    # Load and preprocess input image
    input_example = preprocess_example(sat_path)

    # Predict with the model
    predictions = model.predict(input_example)
    predicted_mask = np.argmax(predictions[0], axis=-1)

    # Load ground truth mask
    gt_mask = np.array(Image.open(gt_path))

    # Visualize the images
    plt.figure(figsize=(15, 5))

    # Input aerial image
    plt.subplot(1, 3, 1)
    plt.imshow(input_example[0])
    plt.title("Aerial Image")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask, cmap="viridis", vmin=0, vmax=num_classes - 1)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap="viridis", vmin=0, vmax=num_classes - 1)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()
    
def visualize_images(sat_folder, gt_folder, model_path=None, num_samples=5):
    # Define colors for each class
    class_colors = {
        0: (0, 0, 0),        # unlabeled
        1: (255, 0, 0),      # buildings
        2: (34, 139, 34),    # woodlands
        3: (0, 0, 255),      # water
        4: (184, 115, 51)    # road
    }

    # Get a list of image files in the folders
    sat_files = os.listdir(sat_folder)
    gt_files = os.listdir(gt_folder)

    # Randomly sample image files
    sampled_files = random.sample(sat_files, min(num_samples, len(sat_files)))

    # Plot pairs or triples of images
    for file in sampled_files:
        sat_image_path = os.path.join(sat_folder, file)
        gt_image_path = os.path.join(gt_folder, file.replace('sat_', 'gt_'))

        # Load images
        sat_image = Image.open(sat_image_path)
        gt_image = Image.open(gt_image_path)

        # Convert ground truth image to numpy array
        gt_array = np.array(gt_image)

        # Apply color mapping for ground truth
        gt_colored = np.zeros(gt_array.shape + (3,), dtype=np.uint8)
        for class_label, color in class_colors.items():
            gt_colored[gt_array == class_label] = np.array(color)

        # Create a subplot with 1 or 2 columns based on the presence of the model
        if model_path:
            # Load the model
            model = keras.models.load_model(model_path)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Predict using the model
            input_image = np.array(sat_image) / 255.0
            input_example = np.expand_dims(input_image, axis=0)
            predictions = model.predict(input_example)
            predicted_mask = np.argmax(predictions[0], axis=-1)

            # Apply color mapping for prediction
            predicted_colored = np.zeros_like(gt_colored)
            for class_label, color in class_colors.items():
                predicted_colored[predicted_mask == class_label] = np.array(color)

            # Plot the predicted image on the right
            axes[2].imshow(predicted_colored)
            axes[2].set_title('Predicted Image')
            axes[2].axis('off')

        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the aerial image on the left
        axes[0].imshow(sat_image)
        axes[0].set_title('Aerial Image')
        axes[0].axis('off')

        # Plot the ground truth image in the middle
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth Image')
        axes[1].axis('off')

        plt.show()
        
def resample_folder(input_folder, output_folder, target_size=(128, 128)):
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
def load_data(sat_folder, gt_folder, batch_size=1, num_classes=5):
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
def simple_model(input_size=(128, 128, 3), num_classes=5):
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

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

# Define U-Net model for 5 classes
def unet_model(input_size=(128, 128, 3), num_classes=5):
    inputs = keras.Input(input_size)
    
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation = "softmax")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model