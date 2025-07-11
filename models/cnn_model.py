import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
  def __init__(self, num_classes=43):
    super(CNNModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout(0.25)
    self.fc1 = nn.Linear(64 * 32 * 32, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.dropout(x)
    x = x.view(x.size(0), -1)  # Flatten
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
