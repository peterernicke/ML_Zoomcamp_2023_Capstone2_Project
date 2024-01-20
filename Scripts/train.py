import os
import numpy as np
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

#from tensorflow.keras.losses import Loss
#from tensorflow.keras.models import Model
#from tensorflow.keras.metrics import MeanIoU
#from sklearn.utils.class_weight import compute_class_weight
#from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose


IMAGE_SIZE = 256
INPUT_SIZE = (IMAGE_SIZE, IMAGE_SIZE, 3)
NUM_CLASSES = 5
TRAIN_EPOCHS = 50
BATCH_SIZE = 1  # You can adjust this based on your available memory

USE_CLASS_WEIGTHS = False
CLASS_WEIGHTS = {0: .59, 1: .32, 2: .06, 3: .02, 4: .01}

TRAIN_SAT_FOLDER = "./../Data50_res256_filtered_20/train/sat/"
TRAIN_GT_FOLDER = "./../Data50_res256_filtered_20/train/gt/"
VAL_SAT_FOLDER = "./../Data50_res256_filtered_20/validate/sat/"
VAL_GT_FOLDER = "./../Data50_res256_filtered_20/validate/gt/"

METRICS = [
      keras.metrics.BinaryCrossentropy(name='cross entropy'),
      keras.metrics.CategoricalCrossentropy(name='cat_cross_entropy'),
      keras.metrics.MeanIoU(name='mIoU', num_classes=NUM_CLASSES),
      keras.metrics.MeanSquaredError(name='Brier score'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

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

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.5)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.5)(x)
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

if __name__ == "__main__":
    # Load and preprocess the training dataset
    X_train, y_train = load_data(TRAIN_SAT_FOLDER, TRAIN_GT_FOLDER, BATCH_SIZE, NUM_CLASSES)

    # Load and preprocess the validation dataset
    X_val, y_val = load_data(VAL_SAT_FOLDER, VAL_GT_FOLDER, BATCH_SIZE, NUM_CLASSES)
    
    # Initialize and compile the model
    model_unet = unet_model(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    model_unet.compile(optimizer="adam", loss="categorical_crossentropy", metrics=METRICS)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        'model_unet_{epoch:02d}_{val_accuracy:.3f}.keras',
        monitor='val_accuracy',     # Monitor validation accuracy
        save_best_only=True,
        save_weights_only=False,    # Save entire model
        mode='max',                 # Save the model with the highest validation accuracy
        verbose=1
    )
    
    # Train the model with class weights and checkpoint callbacks for model saving and early stopping
    if(USE_CLASS_WEIGTHS):
        model_unet.fit(
            X_train, y_train,
            epochs=TRAIN_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            class_weight=CLASS_WEIGHTS,
            callbacks=[checkpoint, early_stopping]
        )
    else:
        # Train the model without class weights but with checkpoint callbacks for model saving and early stopping
        model_unet_history = model_unet.fit(
            X_train, y_train,
            epochs=TRAIN_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            #class_weight=CLASS_WEIGHTS,
            callbacks=[checkpoint, early_stopping]
        )