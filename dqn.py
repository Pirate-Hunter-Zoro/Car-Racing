import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNCNN(nn.Module):
    def __init__(self, action_size):
        super(DQNCNN, self).__init__()
        # We're going to be stacking 4 frames together and gray-scaling, so fhte first convolutional layer will have 4x1=4 input channels
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Dynamically find the size of the first hidden layer
        hidden_size = 0
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 96, 96)  # batch size = 1
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            hidden_size = x.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(hidden_size, 512)
        self.out = nn.Linear(512, action_size)
    
    def forward(self, x):
        assert x.shape[1] == 4, f"Expected 4 stacked frames, got shape {x.shape}"
        x = x / 255.0  # Normalize the pixels
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)