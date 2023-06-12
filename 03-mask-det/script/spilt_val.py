import os
import shutil
import random

input_folder = "images/train"
output_folder = "images/val"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".jpg") or f.endswith(".png")]

num_val_images = int(len(image_paths) * 0.3)

val_image_indexes = random.sample(range(len(image_paths)), num_val_images)

for i in val_image_indexes:
    image_path = image_paths[i]
    shutil.copy(image_path, output_folder)
    os.remove(image_path)