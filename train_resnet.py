import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.custom_dataset import GTSRBCustomDataset
from models.resnet_model import get_resnet_model, print_model_info
import os

# Configuración
BATCH_SIZE = 32
EPOCHS = 15
CSV_PATH = "dataset/Train.csv"
ROOT_DIR = "dataset"

# Estrategia de fine-tuning a usar
STRATEGY = 'partial_finetune'  # Opciones: 'classifier_only', 'partial_finetune', 'full_finetune', 'lightweight'

# Verificar archivos
print("🔍 Checking files...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
if not os.path.exists(ROOT_DIR):
    raise FileNotFoundError(f"Root directory not found: {ROOT_DIR}")

# Verificar CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# Transformaciones optimizadas para ResNet
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño estándar para ResNet
    transforms.RandomRotation(10),  # Augmentation ligero
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Dataset y DataLoader
print("📂 Creating dataset...")
train_dataset = GTSRBCustomDataset(CSV_PATH, ROOT_DIR, transform=transform_train)
print(f"Dataset size: {len(train_dataset)}")

# Probar cargar una muestra
print("🧪 Testing dataset loading...")
try:
    sample_image, sample_label = train_dataset[0]
    print(f"✅ Sample loaded successfully - Image shape: {sample_image.shape}, Label: {sample_label}")
except Exception as e:
    print(f"❌ Error loading sample: {e}")
    exit(1)

# DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0,  # Cambiar a 2-4 cuando funcione bien
    pin_memory=True if device.type == 'cuda' else False,
    drop_last=True
)

# Probar cargar un batch
print("🧪 Testing batch loading...")
try:
    for batch_images, batch_labels in train_loader:
        print(f"✅ Batch loaded successfully - Batch shape: {batch_images.shape}")
        print(f"Labels range: {batch_labels.min().item()} to {batch_labels.max().item()}")
        break
except Exception as e:
    print(f"❌ Error loading batch: {e}")
    exit(1)

# Cargar modelo
print(f"🏗️  Loading ResNet model with strategy: {STRATEGY}")
try:
    model = get_resnet_model(num_classes=43, strategy=STRATEGY).to(device)
    print("✅ ResNet model loaded successfully")
    
    # Mostrar información del modelo
    print_model_info(model)
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Configurar optimizer basado en la estrategia
criterion = nn.CrossEntropyLoss()

if STRATEGY == 'classifier_only':
    # Solo entrenar el clasificador con LR más alto
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad], 
        lr=1e-3, 
        weight_decay=1e-4
    )
    
elif STRATEGY == 'partial_finetune':
    # Diferentes learning rates para diferentes partes
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'layer3' in n and p.requires_grad], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'layer4' in n and p.requires_grad], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'fc' in n and p.requires_grad], 'lr': 1e-3}
    ], weight_decay=1e-4)
    
elif STRATEGY in ['full_finetune', 'lightweight']:
    # Learning rate más bajo para fine-tuning completo
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Scheduler para mejorar convergencia
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

# Crear directorio de salida
os.makedirs("outputs/checkpoints", exist_ok=True)

# Variables para tracking
best_acc = 0.0
patience_counter = 0
max_patience = 5

print(f"🚀 Starting ResNet training...")
print(f"Strategy: {STRATEGY}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Total batches per epoch: {len(train_loader)}")
print(f"Epochs: {EPOCHS}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    batch_count = 0
    
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        try:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            batch_count += 1
            
            # Progreso cada 25 batches
            if (batch_idx + 1) % 25 == 0:
                current_acc = correct / ((batch_idx + 1) * BATCH_SIZE)
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f} - Batch Acc: {current_acc:.4f}")
        
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            continue

    # Estadísticas del epoch
    if batch_count > 0:
        avg_loss = total_loss / batch_count
        epoch_acc = correct / len(train_dataset)
        
        print(f"📊 Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Acc: {epoch_acc:.4f}")
        
        # Step del scheduler
        scheduler.step(epoch_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🎯 Learning rate: {current_lr:.6f}")
        
        # Guardar mejor modelo
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'strategy': STRATEGY
            }, f"outputs/checkpoints/resnet18_best_{STRATEGY}.pth")
            print(f"🏆 New best accuracy: {best_acc:.4f} - Model saved!")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"⏰ Early stopping triggered. No improvement for {max_patience} epochs.")
            break
            
    else:
        print(f"❌ Epoch {epoch+1}/{EPOCHS} - No valid batches processed")

# Guardar modelo final
print("\n💾 Saving final model...")
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': epoch_acc if 'epoch_acc' in locals() else 0,
    'strategy': STRATEGY
}, f"outputs/checkpoints/resnet18_final_{STRATEGY}.pth")

print(f"✅ Modelo ResNet final guardado en outputs/checkpoints/resnet18_final_{STRATEGY}.pth")
print(f"🏆 Mejor modelo guardado en outputs/checkpoints/resnet18_best_{STRATEGY}.pth")

# Resumen final
print(f"\n🏁 Training completed!")
print(f"🎯 Best accuracy achieved: {best_acc:.4f}")
print(f"📊 Strategy used: {STRATEGY}")
print(f"🔄 Total epochs completed: {epoch + 1}")
print(f"📈 Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")