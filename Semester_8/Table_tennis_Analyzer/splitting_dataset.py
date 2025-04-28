import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Get all your image files
image_files = [f for f in os.listdir('extracted_frames') if f.endswith('.jpg')]

# Create random splits (70% train, 15% validation, 15% test)
train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

# Create directories
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/val', exist_ok=True)
os.makedirs('dataset/test', exist_ok=True)

# Copy files to respective directories
for f in train_files:
    shutil.copy(os.path.join('extracted_frames', f), os.path.join('dataset/train', f))
for f in val_files:
    shutil.copy(os.path.join('extracted_frames', f), os.path.join('dataset/val', f))
for f in test_files:
    shutil.copy(os.path.join('extracted_frames', f), os.path.join('dataset/test', f))