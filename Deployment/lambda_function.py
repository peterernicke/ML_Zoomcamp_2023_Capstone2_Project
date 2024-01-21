import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import load_img
import tensorflow.lite as tflite 

model_file = "final-model.tflite"

def get_interpreter():
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    
    return interpreter, input_index, output_index

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

def predict(img):
    # Preprocessing the image
    x = np.array(img)
    x = np.float32(x) / 255.
    # Turning this image into a batch of one image
    X = np.array([x])
    
    interpreter, input_index, output_index = get_interpreter()
    
    # Initializing the input of the interpreter with the image X
    interpreter.set_tensor(input_index, X)

    # Invoking the computations in the neural network
    interpreter.invoke()
    
    # Results are in the output_index. so fetching the results...
    preds = interpreter.get_tensor(output_index)
    
    predicted_mask = np.argmax(preds[0], axis=-1)

    colored_predicted_mask = recolor_prediction(predicted_mask)
            
    PIL_image = Image.fromarray(colored_predicted_mask.astype('uint8'), 'RGB')
    
    return PIL_image
    
def lambda_handler(event, context):
    image_data = event['body'].encode('latin-1')
    
    try:
        PIL_img = predict(image_data)
        
        # Save the image to a BytesIO object
        image_bytes = BytesIO()
        PIL_img.save(image_bytes, format='PNG')
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
