import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", figsize=(10, 8), cmap='Blues', save_path=None):
  plt.figure(figsize=figsize)
  sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
              xticklabels=class_names,
              yticklabels=class_names)
  plt.title(title)
  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')
  plt.xticks(rotation=45)
  plt.tight_layout()
  if save_path:
      plt.savefig(save_path)
  plt.show()

def plot_sample_predictions(images, labels, preds, class_names, n=5):
  plt.figure(figsize=(15, 3))
  for i in range(n):
    image = images[i].permute(1, 2, 0).cpu().numpy()
    image = (image * 0.5 + 0.5).clip(0, 1)  # denormalize
    plt.subplot(1, n, i + 1)
    plt.imshow(image)
    plt.title(f"GT: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
    plt.axis('off')
  plt.tight_layout()
  plt.show()
