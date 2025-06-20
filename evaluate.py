import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.custom_dataset import GTSRBCustomDataset
from models.cnn_model import CNNModel
from utils.metrics import calculate_metrics, print_classification_report, get_confusion_matrix
from utils.plot_utils import plot_confusion_matrix

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = GTSRBCustomDataset(
    csv_path="dataset/Test.csv",
    root_dir="dataset",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64)

model = CNNModel(num_classes=43)
model.load_state_dict(torch.load("outputs/checkpoints/cnn.pth"))
model = model.cuda().eval()

y_true = []
y_pred = []

with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.cuda(), labels.cuda()
    outputs = model(images)
    preds = outputs.argmax(1)

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

# 📊 Métricas
metrics = calculate_metrics(y_true, y_pred)
print("\n📈 Métricas:")
for key, value in metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")

print("\n📋 Classification Report:")
print_classification_report(y_true, y_pred)

cm = get_confusion_matrix(y_true, y_pred)

class_names = [str(i) for i in range(43)]

plot_confusion_matrix(cm, class_names=class_names, title="Confusion Matrix")
