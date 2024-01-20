import os
import shutil
from PIL import Image
from tqdm import tqdm

# Set the size threshold for classifying images as big or small
SIZE_THRESHOLD = 64000000  # 8.000*8.000

# Define source and destination folders
src_folder = "./../laco.ai"
dest_folder_big = "./../laco.ai_big"
dest_folder_small = "./../laco.ai_small"

def divide_and_copy_images(src_folder, dest_folder_small, dest_folder_big, size_threshold):
    image_folder = os.path.join(src_folder, "images")
    mask_folder = os.path.join(src_folder, "masks")

    dest_image_folder_small = os.path.join(dest_folder_small, "images")
    dest_mask_folder_small = os.path.join(dest_folder_small, "masks")

    os.makedirs(dest_image_folder_small, exist_ok=True)
    os.makedirs(dest_mask_folder_small, exist_ok=True)
    
    dest_image_folder_big = os.path.join(dest_folder_big, "images")
    dest_mask_folder_big = os.path.join(dest_folder_big, "masks")

    os.makedirs(dest_image_folder_big, exist_ok=True)
    os.makedirs(dest_mask_folder_big, exist_ok=True)

    filenames = os.listdir(image_folder)

    for filename in tqdm(filenames, desc="Copying files", unit="file"):
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        # Open the image to get its size
        with Image.open(image_path) as img:
            width, height = img.size

        # Determine whether the image is big or small based on the size_threshold
        if width * height > size_threshold:
            dest_image_path = os.path.join(dest_image_folder_big, filename)
            dest_mask_path = os.path.join(dest_mask_folder_big, filename)
            shutil.copy(image_path, dest_image_path)
            shutil.copy(mask_path, dest_mask_path)
        else:
            dest_image_path = os.path.join(dest_image_folder_small, filename)
            dest_mask_path = os.path.join(dest_mask_folder_small, filename)
            shutil.copy(image_path, dest_image_path)
            shutil.copy(mask_path, dest_mask_path)

if __name__ == "__main__":

    # Resize and copy for laco.ai_small and laco.ai_big
    divide_and_copy_images(src_folder, dest_folder_small, dest_folder_big, SIZE_THRESHOLD)
