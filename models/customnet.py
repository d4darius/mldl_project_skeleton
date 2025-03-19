from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        # (input, output, kernel size over the image, padding)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*28*28, 200)
        self.relu = nn.ReLU()


    def forward(self, x):
        # Define forward pass
        x = self.conv1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.conv3(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)

        return x