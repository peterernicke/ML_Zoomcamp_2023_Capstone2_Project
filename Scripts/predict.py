import numpy as np

from PIL import Image
from io import BytesIO
from tensorflow import keras
from flask import Flask, request, jsonify, send_file

# Function to load and preprocess a single example
def preprocess_example(img):
    input_image = np.array(Image.open(img)) / 255.0
    #input_image = np.array(img) / 255.0
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

app = Flask('Landcover Segmentation for Aerial Images')
model_path = "./../Model/model_unet_noClassWeights_drop50_256_30_0.972.keras"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    if 'image'not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    imagefile = request.files['image']
    
    # Check if the file has an allowed extension
    allowed_extensions = {'tif', 'tiff'}
    if '.' in imagefile.filename and imagefile.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file extension"}), 400
    
    try:
        # Load the model
        model = keras.models.load_model(model_path)
    
        # Load and preprocess input image
        input_image = preprocess_example(imagefile)
    
        # Predict with the model
        predictions = model.predict(input_image)
        predicted_mask = np.argmax(predictions[0], axis=-1)

        colored_predicted_mask = recolor_prediction(predicted_mask)
        
        PIL_image = Image.fromarray(colored_predicted_mask.astype('uint8'), 'RGB')
        
        # Save the image to a BytesIO object
        image_bytes = BytesIO()
        PIL_image.save(image_bytes, format='PNG')
        
        # Send the image back to the client
        image_bytes.seek(0)
        return send_file(image_bytes, mimetype='image/png', as_attachment=True, download_name='predicted_image.png')
         
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9797)