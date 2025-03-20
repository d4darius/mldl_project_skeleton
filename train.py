from models.customnet import CustomNet
from utils.func import train
from data.dataloader import train_loader
import os
from datetime import datetime
import torch
from torch import nn

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train_accuracy = train(epoch, model, train_loader, criterion, optimizer)

    # Best validation accuracy
    best_acc = max(best_acc, train_accuracy)


print(f'Best Training accuracy: {best_acc:.2f}%')

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# Generate a filename with the current timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'checkpoints/model_{timestamp}.pth'

# Save the model's state dictionary
torch.save(model.state_dict(), filename)