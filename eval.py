from utils.func import validate
from data.dataloader import val_loader
import torch
import os

# Load the latest model checkpoint from the checkpoints folder
checkpoint_dir = 'checkpoints'
latest_checkpoint = max(
    [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
    key=os.path.getctime
)

# Initialize the model and load the state dictionary
model = CustomEnet().cuda()
model.load_state_dict(torch.load(latest_checkpoint))
model.eval()
criterion = nn.CrossEntropyLoss()

val_accuracy = validate(model, val_loader, criterion)