from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
import os

transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset_path = os.path.join(os.getcwd(), 'dataset')

tiny_imagenet_dataset_train = ImageFolder(root=os.path.join(dataset_path, 'tiny-imagenet-200', 'train'), transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root=os.path.join(dataset_path, 'tiny-imagenet-200', 'val'), transform=transform)

print("Loading the dataset...")
print(f"Training samples: {len(tiny_imagenet_dataset_train)}")
train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)
print(f"Validation samples: {len(tiny_imagenet_dataset_val)}")
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)