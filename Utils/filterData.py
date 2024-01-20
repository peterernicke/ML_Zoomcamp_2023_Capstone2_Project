import os
import shutil
import numpy as np
from PIL import Image

def filter_data_by_class(folder, class_name, threshold_percent):
    sat_folder = os.path.join(folder, "sat")
    gt_folder = os.path.join(folder, "gt")

    filtered_sat_files = []
    filtered_gt_files = []

    for filename in os.listdir(gt_folder):
        if filename.endswith(".tif") and filename.startswith("gt_"):
            gt_path = os.path.join(gt_folder, filename)
            sat_path = os.path.join(sat_folder, filename.replace("gt_", "sat_"))

            # Load ground truth mask
            gt_mask = np.array(Image.open(gt_path))

            # Calculate the percentage of the specified class
            class_percent = np.sum(gt_mask == class_name) / gt_mask.size * 100.0

            # Check if the class percentage is below the threshold
            if class_percent <= threshold_percent:
                filtered_gt_files.append(gt_path)
                filtered_sat_files.append(sat_path)

    return filtered_sat_files, filtered_gt_files

def copy_files_to_folder_structure(filtered_sat_files, filtered_gt_files, destination_folder):
    os.makedirs(os.path.join(destination_folder, "sat"), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "gt"), exist_ok=True)

    for source_file in filtered_sat_files:
        destination_path = os.path.join(destination_folder, "sat", os.path.basename(source_file))
        shutil.copyfile(source_file, destination_path)

    for source_file in filtered_gt_files:
        destination_path = os.path.join(destination_folder, "gt", os.path.basename(source_file))
        shutil.copyfile(source_file, destination_path)