import os
import shutil
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# Load image paths and store in a numpy array, images.
images = glob.glob("Cityscape Dataset/*/*")
images = np.array(images)

# Load the corresponding labels by splitting the file path on the delimiter.
# Take the second from last entry in the array as the label (black, blue, green, other).
labels = np.array([f.split("\\")[-2] for f in images])

# Split the data into train, validation, and test sets
train_images, test_val_images, train_labels, test_val_labels = train_test_split(images, labels, test_size=0.2, random_state=1)
val_images, test_images, val_labels, test_labels = train_test_split(test_val_images, test_val_labels, test_size=0.5, random_state=1)

# Create directories for train, validation, and test sets
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Move images to train directory
for image, label in zip(train_images, train_labels):
    label_dir = os.path.join("data/train", label)
    os.makedirs(label_dir, exist_ok=True)
    shutil.move(image, label_dir)

# Move images to validation directory
for image, label in zip(val_images, val_labels):
    label_dir = os.path.join("data/val", label)
    os.makedirs(label_dir, exist_ok=True)
    shutil.move(image, label_dir)

# Move images to test directory
for image, label in zip(test_images, test_labels):
    label_dir = os.path.join("data/test", label)
    os.makedirs(label_dir, exist_ok=True)
    shutil.move(image, label_dir)

print("Data moved successfully.")