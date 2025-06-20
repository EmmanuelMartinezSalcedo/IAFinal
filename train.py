import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.custom_dataset import GTSRBCustomDataset
from models.cnn_model import CNNModel

transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor(),
  transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = GTSRBCustomDataset("dataset/Train.csv", root_dir="dataset", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = CNNModel(num_classes=43).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
  model.train()
  total_loss = 0
  correct = 0
  for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    correct += (outputs.argmax(1) == labels).sum().item()

  acc = correct / len(train_dataset)
  print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")
