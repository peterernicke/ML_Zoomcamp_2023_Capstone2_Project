import os
import random
import rasterio
import numpy as np
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# Randomly sample ground truth files
def random_sample_files(folder, num_samples):
    all_files = os.listdir(folder)
    return random.sample(all_files, min(num_samples, len(all_files)))

# Visualize class distribution from a subset of ground truth files
def visualize_class_distribution(folder, num_samples=100, title='Class Distribution'):
    # Mapping from class labels to class names
    class_names = {
        0: 'unlabeled',
        1: 'buildings',
        2: 'woodlands',
        3: 'water',
        4: 'road'
    }

    # Mapping from class names to custom colors
    class_colors = {
        'unlabeled': 'gray',
        'buildings': 'red',
        'woodlands': 'forestgreen',
        'water': 'blue',
        'road': 'brown'
    }

    sampled_files = random_sample_files(folder, num_samples)
    class_co_occurrence_matrix = np.zeros((len(class_names), len(class_names)))

    class_counts = {}

    for file in tqdm(sampled_files, desc="Processing files", unit="file"):
        file_path = os.path.join(folder, file)

        with rasterio.open(file_path) as dataset:
            image = dataset.read(1)
        
        flattened_image = image.flatten()
        unique_classes, class_label_counts = zip(*[(label, list(flattened_image).count(label)) for label in set(flattened_image)])

        for class_label, count in zip(unique_classes, class_label_counts):
            # Convert class label to class name
            class_name = class_names.get(class_label, f'Class_{class_label}')
            
            if class_name not in class_counts:
                class_counts[class_name] = count
            else:
                class_counts[class_name] += count
        
        if len(unique_classes) > 1:
            # Update the co-occurrence matrix for each pair of classes
            for i in range(len(unique_classes)):
                for j in range(i + 1, len(unique_classes)):
                    class_i = class_names.get(unique_classes[i], f'Class_{unique_classes[i]}')
                    class_j = class_names.get(unique_classes[j], f'Class_{unique_classes[j]}')

                    class_co_occurrence_matrix[i, j] += class_label_counts[i]
                    class_co_occurrence_matrix[j, i] += class_label_counts[j]

    total_samples = sum(class_counts.values())
    class_frequencies = [count / total_samples for count in class_counts.values()]

    # Plot the class distribution bar chart with custom colors
    colors = [class_colors.get(class_name, 'gray') for class_name in class_counts.keys()]
    plt.bar(class_counts.keys(), class_frequencies, color=colors)
    plt.xlabel('Class Label')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

    # Pie chart with custom colors
    pie_colors = [class_colors.get(class_name, 'gray') for class_name in class_counts.keys()]
    plt.pie(class_frequencies, labels=class_counts.keys(), autopct='%1.1f%%', colors=pie_colors)
    plt.title('Class Distribution Pie Chart')
    plt.show()

    # Class distribution summary
    print("Class Distribution Summary:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} samples, {count/total_samples:.2%}")
    
    # Create a heatmap for class relationships with a logarithmic scale
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.log1p(class_co_occurrence_matrix), annot=True, cmap='Blues', xticklabels=class_names.values(), yticklabels=class_names.values())
    plt.xlabel('Class Label')
    plt.ylabel('Class Label')
    plt.title('Co-Occurrence of Classes')
    plt.show()

def visualize_image_pairs(sat_folder, gt_folder, num_samples=5):
    # Define colors for each class
    class_colors = {
        0: (0, 0, 0),        # unlabeled
        1: (255, 0, 0),      # buildings
        2: (34, 139, 34),    # woodlands
        3: (0, 0, 255),      # water
        4: (184, 115, 51)    # road
    }

    # Get a list of image files in the folders
    sat_files = os.listdir(sat_folder)
    gt_files = os.listdir(gt_folder)

    # Randomly sample image files
    sampled_files = random.sample(sat_files, min(num_samples, len(sat_files)))

    # Plot pairs of images
    for file in sampled_files:
        sat_image_path = os.path.join(sat_folder, file)
        gt_image_path = os.path.join(gt_folder, file.replace('sat_', 'gt_'))

        # Load images
        sat_image = Image.open(sat_image_path)
        gt_image = Image.open(gt_image_path)

        # Convert ground truth image to numpy array
        gt_array = np.array(gt_image)

        # Apply color mapping
        gt_colored = np.zeros(gt_array.shape + (3,), dtype=np.uint8)
        for class_label, color in class_colors.items():
            gt_colored[gt_array == class_label] = np.array(color)

        # Create a subplot with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the satellite image on the left
        axes[0].imshow(sat_image)
        axes[0].set_title('Satellite Image')
        axes[0].axis('off')

        # Plot the ground truth image on the right
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth Image')
        axes[1].axis('off')

        plt.show()
       
def resample_images(input_folder, output_folder, target_size=(256, 256)):
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
        #Image.ANTIALIAS: Anti-aliasing to reduce artifacts
        #Image.NEAREST: Nearest-neighbor sampling
        #Image.BOX: Box sampling
        #Image.BILINEAR: Bilinear interpolation
        #Image.HAMMING: Hamming-windowed sinc interpolation
        #Image.BICUBIC: Bicubic interpolation
        #Image.LANCZOS: Lanczos-windowed sinc interpolation
        resampled_img = img.resize(target_size, Image.ANTIALIAS)

        # Save the resampled image
        resampled_img.save(output_path)

    print("Resampling completed.")        
        
        
        
        