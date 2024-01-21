import requests
from PIL import Image
from io import BytesIO

url = 'http://localhost:9797/predict'

# Load the TIFF image from file or any other source
image_path = "./../Data50_res256_filtered_20/validate/sat/sat_N-33-60-D-d-1-2_02_01.tif"
with open(image_path, 'rb') as file:
    image_data = file.read()

# Send the POST request
response = requests.post(url, files={'image': image_data}).content

# Convert the response content (image bytes) to a PIL Image and save it
predicted_image = Image.open(BytesIO(response))

# Save the predicted image locally
predicted_image.save("./predicted_image.png")

#print("Prediction for the provided TIFF image:")
#print(response)