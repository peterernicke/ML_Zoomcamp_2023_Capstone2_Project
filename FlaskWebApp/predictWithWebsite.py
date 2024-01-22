import os
import uuid
import numpy as np

from PIL import Image
from io import BytesIO
from tensorflow import keras
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory, url_for


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

# Set the path to the uploads folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

# Configure the upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the uploaded file with a unique identifier
        file_id = str(uuid.uuid4())
        tif_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.tif')
        png_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.png')
        pred_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_pred.png')

        file.save(tif_path)

        # Convert TIFF to PNG
        img = Image.open(tif_path)
        img.save(png_path)

        try:
            # Load the model
            model = keras.models.load_model(model_path)

            # Load and preprocess input image
            input_image = preprocess_example(png_path)

            # Predict with the model
            predictions = model.predict(input_image)
            predicted_mask = np.argmax(predictions[0], axis=-1)

            colored_predicted_mask = recolor_prediction(predicted_mask)

            PIL_image = Image.fromarray(colored_predicted_mask.astype('uint8'), 'RGB')

            # Save the predicted image to a BytesIO object
            PIL_image.save(pred_path)

            # Send the predicted image back to the client
            return render_template('result.html', original_image=f'{file_id}.png',
                                        predicted_image=f'{file_id}_pred.png')

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9898)

