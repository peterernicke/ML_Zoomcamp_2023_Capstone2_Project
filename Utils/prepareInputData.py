import os
import random
from PIL import Image
from tqdm import tqdm

# Main folder of input files - This is the unzipped folder from https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip
main_folder = './../laco.ai'

################################################
##############                    ##############
############   Constants section    ############
##############                    ##############

SOURCE_IMAGES_FOLDER = os.path.join(main_folder, 'images')
SOURCE_MASKS_FOLDER = os.path.join(main_folder, 'masks')
PREFIX_SAT = 'sat_'
PREFIX_GT = 'gt_'

# Tile size for cropping
TILE_SIZE = 512

# Ratios for dataset splitting
TRAIN_RATIO = 0.8
VALIDATE_RATIO = 0.1

# Path to the output folders
DATA_FOLDER = './../Data'
TRAIN_SAT_FOLDER = os.path.join(DATA_FOLDER, 'train/sat')
TRAIN_GT_FOLDER = os.path.join(DATA_FOLDER, 'train/gt')
VALIDATE_SAT_FOLDER = os.path.join(DATA_FOLDER, 'validate/sat')
VALIDATE_GT_FOLDER = os.path.join(DATA_FOLDER, 'validate/gt')
TEST_SAT_FOLDER = os.path.join(DATA_FOLDER, 'test/sat')
TEST_GT_FOLDER = os.path.join(DATA_FOLDER, 'test/gt')

# Output folder structure
FOLDER_STRUCTURE = [
    'train',
    'validate',
    'test',
    'train/sat',
    'train/gt',
    'validate/sat',
    'validate/gt',
    'test/sat',
    'test/gt',
]

################################################
##############                    ##############
############    Function section    ############
##############                    ##############

def create_folders(data_folder, folders):
    """
        Create folders and subfolders.

        Parameters:
        - data_folder (str): Path to the main folder.
        - folders (list): List of folder names to be created.
    """
    for folder in folders:
        folder_path = os.path.join(data_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

def split_dataset(image_files, mask_files, train_ratio=0.8, validate_ratio=0.1):
    """
        Split the dataset into train, validate, and test sets.

        Parameters:
        - image_files (list): List of aerial image file names.
        - mask_files (list): List of mask image file names.
        - train_ratio (float): Ratio of files for training (default: 0.8).
        - validate_ratio (float): Ratio of files for validation (default: 0.1).

        Returns:
        - train_files (list): List of file pairs (sat,ground truth) for training.
        - validate_files (list): List of file pairs (sat,ground truth) for validation.
        - test_files (list): List of file pairs (sat,ground truth) for testing.
    """

    file_pairs = list(zip(image_files, mask_files))

    random.shuffle(file_pairs)

    # Calculate the split indices for train, validate, and test
    total_files = len(file_pairs)
    train_split = int(train_ratio * total_files)
    validate_split = int(validate_ratio * total_files)

    # Split the file pairs into train, validate, and test sets
    train_files = file_pairs[:train_split]
    validate_files = file_pairs[train_split:train_split + validate_split]
    test_files = file_pairs[train_split + validate_split:]

    return train_files, validate_files, test_files

def crop_image(image_path, output_folder, tile_size, prefix):
    """
        Crop an image and save it as subtiles of shape (tile_size, tile_size).

        Parameters:
        - image_path (str): Path to the input image.
        - output_folder (str): Path to the output folder.
        - tile_size (int): Size (height and width) of each tile.
        - prefix (str): Prefix for the output tile names.
    """

    img = Image.open(image_path)

    # Get the dimensions of the image
    width, height = img.size

    # Calculate the number of tiles in each dimension
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    # Get the file name without extension
    filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]

    # Iterate over each tile
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Calculate the coordinates for each tile
            left = i * tile_size
            upper = j * tile_size
            right = left + tile_size
            lower = upper + tile_size

            # Crop the image to the tile
            tile = img.crop((left, upper, right, lower))

            # Save the tile with a unique name
            tile_filename = f"{prefix}{filename_without_extension}_{i+1:02d}_{j+1:02d}.tif"
            tile_path = os.path.join(output_folder, tile_filename)

            tile.save(tile_path)

def copy_and_crop_files(file_list, destination_sat_folder, destination_gt_folder, progress_desc):
    """
        Crop image and mask files and copy them to the destination folders.

        Parameters:
        - file_list (list): List of file pairs.
        - destination_sat_folder (str): Path to the destination sat folder.
        - destination_gt_folder (str): Path to the destination gt folder.
        - progress_desc (str): Description for the tqdm progress bar.
    """
    for image_file, mask_file in tqdm(file_list, desc=progress_desc, unit="file"):
        # Crop image and copy to sat folder
        crop_image(os.path.join(SOURCE_IMAGES_FOLDER, image_file), destination_sat_folder, TILE_SIZE, PREFIX_SAT)

        # Crop mask and copy to gt folder
        crop_image(os.path.join(SOURCE_MASKS_FOLDER, mask_file), destination_gt_folder, TILE_SIZE, PREFIX_GT)


################################################
##############                    ##############
############      Main section      ############
##############                    ##############

if __name__ == "__main__":        
    # Create the folders and subfolders
    create_folders(DATA_FOLDER, FOLDER_STRUCTURE)
    print("Folders and subfolders created.")

    # Get the list of image and mask files
    image_files = os.listdir(SOURCE_IMAGES_FOLDER)
    mask_files = os.listdir(SOURCE_MASKS_FOLDER)

    # Split dataset into datasets for train, validate and test
    train_files, validate_files, test_files = split_dataset(image_files, 
                                                            mask_files, 
                                                            train_ratio=TRAIN_RATIO, 
                                                            validate_ratio=VALIDATE_RATIO)

    # Copy and resize (image, mask) pairs to the train folders
    copy_and_crop_files(train_files, 
                          TRAIN_SAT_FOLDER, 
                          TRAIN_GT_FOLDER, 
                          progress_desc="Copying and cropping training files")

    # Copy and resize (image, mask) pairs to the validate folders
    copy_and_crop_files(validate_files, 
                          VALIDATE_SAT_FOLDER, 
                          VALIDATE_GT_FOLDER, 
                          progress_desc="Copying and cropping validation files")

    # Copy and resize (image, mask) pairs to the test folders
    copy_and_crop_files(test_files, 
                          TEST_SAT_FOLDER, 
                          TEST_GT_FOLDER, 
                          progress_desc="Copying and cropping test files")

    print("Data copying and splitting completed.")