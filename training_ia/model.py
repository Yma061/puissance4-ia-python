import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Net(nn.Module):

    def __init__(self):
        super().__init__()

        # ===== Convolution =====
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # ===== Fully connected =====
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):

        # x shape : (batch, 6, 7)
        x = x.unsqueeze(1)  # (batch, 1, 6, 7)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)