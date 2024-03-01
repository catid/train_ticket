import torch
import torch.nn as nn
import torch.nn.functional as F

from .ticket_linear import TicketLinear

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)  # Input size assuming 28x28 images
        self.fc2 = TicketLinear(512, 256, seed=42)
        self.fc3 = TicketLinear(256, 128, seed=50)
        self.fc4 = nn.Linear(128, 10)   # Output 10 classes

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No need for dropout in the simple MLP version
        return x
