import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.custom_dataset import GTSRBCustomDataset
from models.cnn_model import CNNModel
import os

# Configuración
BATCH_SIZE = 16
EPOCHS = 10
CSV_PATH = "dataset/Train.csv"
ROOT_DIR = "dataset"

# Verificar que los archivos existan
print("Checking files...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
if not os.path.exists(ROOT_DIR):
    raise FileNotFoundError(f"Root directory not found: {ROOT_DIR}")

# Verificar CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset y DataLoader
print("Creating dataset...")
train_dataset = GTSRBCustomDataset(CSV_PATH, ROOT_DIR, transform=transform)
print(f"Dataset size: {len(train_dataset)}")

# Probar cargar una muestra
print("Testing dataset loading...")
try:
    sample_image, sample_label = train_dataset[0]
    print(f"Sample loaded successfully - Image shape: {sample_image.shape}, Label: {sample_label}")
except Exception as e:
    print(f"Error loading sample: {e}")
    exit(1)

# DataLoader con num_workers=0 para evitar problemas de multiprocessing
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0,  # Importante para debugging
    pin_memory=True if device.type == 'cuda' else False
)

# Probar cargar un batch
print("Testing batch loading...")
try:
    for batch_images, batch_labels in train_loader:
        print(f"Batch loaded successfully - Batch shape: {batch_images.shape}")
        break
except Exception as e:
    print(f"Error loading batch: {e}")
    exit(1)

# Modelo
print("Loading model...")
model = CNNModel(num_classes=43).to(device)
print(f"Model loaded on {device}")

# Loss y optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Crear directorio de salida
os.makedirs("outputs/checkpoints", exist_ok=True)

# Entrenamiento
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    batch_count = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        try:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            batch_count += 1
            
            # Progreso cada 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    if batch_count > 0:
        avg_loss = total_loss / batch_count
        acc = correct / len(train_dataset)
        print(f"📚 Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")
    else:
        print(f"❌ Epoch {epoch+1}/{EPOCHS} - No valid batches processed")

# Guardar modelo
print("Saving model...")
torch.save(model.state_dict(), "outputs/checkpoints/cnn.pth")
print("✅ Modelo CNN guardado en outputs/checkpoints/cnn.pth")