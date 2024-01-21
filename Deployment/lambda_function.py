import os
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow import keras
from tempfile import NamedTemporaryFile

# Function to load and preprocess a single example
def preprocess_example(img):
    input_image = np.array(Image.open(img)) / 255.0
    return np.expand_dims(input_image, axis=0)

def recolor_prediction(predicted_mask):
    # Define colors for each class
    class_colors = {
        0: (0, 0, 0),        # unlabeled
        1: (255, 0, 0),      # buildings
        2: (34, 139, 34),    # woodlands
        3: (0, 0, 255),      # water
        4: (184, 115, 51)    # road
    }

    # Convert ground truth image to numpy array
    pred_array = np.array(predicted_mask)

    # Apply color mapping
    pred_colored = np.zeros(pred_array.shape + (3,), dtype=np.uint8)
    for class_label, color in class_colors.items():
        pred_colored[pred_array == class_label] = np.array(color)
    
    return pred_colored

def lambda_handler(event, context):
    image_data = event['body'].encode('latin-1')
    try:
        # Load the model
        model_path = "final-model.keras"  # Update with your model path
        model = keras.models.load_model(model_path)

        # Load and preprocess input image
        input_image = preprocess_example(BytesIO(image_data))

        # Predict with the model
        predictions = model.predict(input_image)
        predicted_mask = np.argmax(predictions[0], axis=-1)

        colored_predicted_mask = recolor_prediction(predicted_mask)

        PIL_image = Image.fromarray(colored_predicted_mask.astype('uint8'), 'RGB')

        # Save the image to a BytesIO object
        image_bytes = BytesIO()
        PIL_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # Read the BytesIO object content and convert to bytes
        image_content_bytes = image_bytes.read()

        return {
            'statusCode': 200,
            'body': image_content_bytes.decode('latin-1'),
            'headers': {
                'Content-Type': 'image/png',
                'Content-Disposition': 'attachment; filename=predicted_image.png'
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        }
