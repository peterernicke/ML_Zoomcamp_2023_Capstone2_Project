import os
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

# Function to load and preprocess the dataset
def load_data(sat_folder, gt_folder, num_classes, batch_size=1):
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

# deeplabv3_plus architecture found on https://keras.io/examples/vision/deeplabv3_plus/
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
        activation="relu"
    )(block_input)
    x = layers.BatchNormalization()(x)
    #return ops.nn.relu(x)
    return x

# deeplabv3_plus architecture found on https://keras.io/examples/vision/deeplabv3_plus/
def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
        #interpolation="lanczos",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

# deeplabv3_plus architecture found on https://keras.io/examples/vision/deeplabv3_plus/
def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    
# Function to load and preprocess a batch of the dataset
def load_batch(sat_folder, gt_folder, filenames, target_size=(128, 128), num_classes=5):
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
def evaluate_model_on_folder(model, sat_folder, gt_folder, target_size=(128, 128)):
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

        # Plot the satellite image on the left
        axes[0].imshow(sat_image)
        axes[0].set_title('Aerial Image')
        axes[0].axis('off')

        # Plot the ground truth image in the middle
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth Image')
        axes[1].axis('off')

        plt.show() 