import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
from collections import defaultdict


def download_and_extract_cifar10(root='./data'):
    """
    Download and extract CIFAR-10 dataset if it doesn't exist.
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    if not os.path.exists(root):
        os.makedirs(root)

    filepath = os.path.join(root, filename)

    if not os.path.exists(filepath):
        print("Downloading CIFAR-10...")
        urlretrieve(url, filepath)

    extract_path = os.path.join(root, 'cifar-10-batches-py')
    if not os.path.exists(extract_path):
        print("Extracting files...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=root)

    return extract_path


def load_batch(file_path):
    """
    Load a single CIFAR-10 batch file.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def load_balanced_cifar10(samples_per_class=100, root='./data', train=True):
    """
    Load a balanced subset of CIFAR-10 images into a dictionary.

    Args:
        samples_per_class (int): Number of samples to load per class
        root (str): Root directory to store/load CIFAR-10 data
        train (bool): Whether to load from training or test set

    Returns:
        dict: Dictionary with class names as keys and lists of numpy arrays (3x32x32) as values
    """
    # Define the class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Download and extract the dataset if needed
    data_path = download_and_extract_cifar10(root)

    # Initialize dictionary to store images by class
    class_images = defaultdict(list)
    class_counts = {name: 0 for name in class_names}

    if train:
        # Load training batches (1-5)
        batch_files = [f'data_batch_{i}' for i in range(1, 6)]
    else:
        # Load test batch
        batch_files = ['test_batch']

    # Process each batch file
    for batch_file in batch_files:
        batch_path = os.path.join(data_path, batch_file)
        batch = load_batch(batch_path)

        # Get images and labels
        images = batch[b'data']
        labels = batch[b'labels']

        # Process each image
        for img, label in zip(images, labels):
            class_name = class_names[label]

            # If we haven't collected enough samples for this class
            if class_counts[class_name] < samples_per_class:
                # Reshape from (3072,) to (32, 32, 3)
                img_reshaped = img.reshape(3, 32, 32).transpose(1, 2, 0)

                # Add to our collection
                class_images[class_name].append(img_reshaped)
                class_counts[class_name] += 1

            # Check if we've collected enough samples for all classes
            if all(count >= samples_per_class for count in class_counts.values()):
                break

        # If we have enough samples, stop processing batches
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    # Convert defaultdict to regular dict
    return dict(class_images)


# Example usage:
"""
# Load 100 images per class from the training set
balanced_cifar = load_balanced_cifar10(samples_per_class=100)

# Access images for a specific class
airplanes = balanced_cifar['airplane']  # List of 100 numpy arrays (3x32x32)

# Print shapes to verify
for class_name, images in balanced_cifar.items():
    print(f"{class_name}: {len(images)} images, shape: {images[0].shape}")
"""

balanced_cifar = load_balanced_cifar10(samples_per_class=100)

import matplotlib.pyplot as plt

airplane = balanced_cifar['airplane'][1]

# Display the image
plt.figure(figsize=(3, 3))
plt.imshow(airplane)
plt.axis('off')
plt.show()
