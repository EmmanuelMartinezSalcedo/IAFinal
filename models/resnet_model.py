import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_resnet_model(num_classes=43, strategy='partial_finetune'):
  if strategy == 'lightweight':
    return get_lightweight_resnet(num_classes)
  
  model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
  
  model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.4),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes)
  )
  
  if strategy == 'classifier_only':
    for param in model.parameters():
      param.requires_grad = False
    for param in model.fc.parameters():
      param.requires_grad = True
          
  elif strategy == 'partial_finetune':
    for param in model.parameters():
      param.requires_grad = False
    
    for param in model.layer3.parameters():
      param.requires_grad = True
    for param in model.layer4.parameters():
      param.requires_grad = True
    for param in model.fc.parameters():
      param.requires_grad = True
          
  elif strategy == 'full_finetune':
    for param in model.parameters():
      param.requires_grad = True
  
  return model

def get_lightweight_resnet(num_classes=43):
  base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
  
  features = nn.Sequential(
    base_model.conv1,
    base_model.bn1,
    base_model.relu,
    base_model.maxpool,
    base_model.layer1,
    base_model.layer2,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
  )
  
  classifier = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
  )
  
  model = nn.Sequential(features, classifier)
  
  return model

def print_model_info(model):
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  print(f"Model Information:")
  print(f"Total parameters: {total_params:,}")
  print(f"Trainable parameters: {trainable_params:,}")
  print(f"Frozen parameters: {total_params - trainable_params:,}")
  print(f"Trainable ratio: {trainable_params/total_params:.2%}")
  
  return total_params, trainable_params