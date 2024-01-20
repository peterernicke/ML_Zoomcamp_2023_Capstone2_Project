import os
from PIL import Image
from tqdm import tqdm

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
        resampled_img = img.resize(target_size, Image.LANCZOS)

        # Save the resampled image
        resampled_img.save(output_path)

    print(f"Resampling completed for {input_folder}.")