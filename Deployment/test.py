import requests
import json
from PIL import Image
from io import BytesIO

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# Load the TIFF image from file or any other source
image_path = "./../Data50_res256_filtered_20/validate/sat/sat_M-33-48-A-c-4-4_01_09.tif"
with open(image_path, 'rb') as file:
    image_data = file.read()

# Create a JSON payload with the image data
payload = {
    'body': image_data.decode('latin-1')  # Convert binary data to string
}

# Send the POST request to your Lambda endpoint
response = requests.post(url, data=json.dumps(payload))

# Check if the request was successful
if response.status_code == 200:
    # Convert the response content (image bytes) to a PIL Image and save it
    predicted_image = Image.open(BytesIO(response.content))
    
    # Save the predicted image locally
    predicted_image.save("./predicted_image.png")
    print("Prediction successful. Image saved.")
else:
    print("Error:", response.text)
