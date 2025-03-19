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
        zip_file.extractall('../dataset')
    print('Download and extraction complete!')

with open('../dataset/tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'../dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'../dataset/tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'../dataset/tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('../dataset/tiny-imagenet/tiny-imagenet-200/val/images')