import requests
from zipfile import ZipFile
from io import BytesIO
import os
import shutil

# Define the path to the dataset
dataset_path = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

# Send a GET request to the URL
response = requests.get(dataset_path)
# Check if the request was successful
if response.status_code == 200:
    # Open the downloaded bytes and extract them
    with ZipFile(BytesIO(response.content)) as zip_file:
        # Construct the path relative to the current working directory
        dataset_path = os.path.join(os.getcwd(), 'dataset')
        zip_file.extractall(dataset_path)
    print('Download and extraction complete!')

val_annotations_path = os.path.join(dataset_path, 'tiny-imagenet-200', 'val', 'val_annotations.txt')
val_images_path = os.path.join(dataset_path, 'tiny-imagenet-200', 'val', 'images')

with open(val_annotations_path) as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        cls_dir = os.path.join(dataset_path, 'tiny-imagenet-200', 'val', cls)
        os.makedirs(cls_dir, exist_ok=True)

        shutil.copyfile(
            os.path.join(val_images_path, fn),
            os.path.join(cls_dir, fn)
        )

shutil.rmtree(val_images_path)