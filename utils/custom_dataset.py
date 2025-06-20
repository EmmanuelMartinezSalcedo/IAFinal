import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset

class GTSRBCustomDataset(Dataset):
  def __init__(self, csv_path, root_dir, transform=None, use_roi=True):
    print(f"Loading CSV from: {csv_path}")
    self.data = pd.read_csv(csv_path)
    self.root_dir = root_dir
    self.transform = transform
    self.use_roi = use_roi
    
    required_columns = ["Path", "ClassId"]
    if self.use_roi:
      required_columns.extend(["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"])
    
    missing_columns = [col for col in required_columns if col not in self.data.columns]
    if missing_columns:
      raise ValueError(f"Missing columns in CSV: {missing_columns}")
    
    print(f"Dataset loaded: {len(self.data)} samples")
    print(f"Columns: {list(self.data.columns)}")
    
    self._validate_paths()

  def _validate_paths(self):
    print("Validating image paths...")
    for i in range(min(10, len(self.data))):
      row = self.data.iloc[i]
      image_path = os.path.join(self.root_dir, row["Path"])
      if not os.path.exists(image_path):
        print(f"WARNING: Image not found: {image_path}")
      else:
        print(f"✓ Image {i+1} exists: {row['Path']}")

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    try:
      row = self.data.iloc[idx]
      image_path = os.path.join(self.root_dir, row["Path"])
      
      if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

      label = int(row["ClassId"])
      
      image = Image.open(image_path).convert("RGB")
      
      if self.use_roi:
        x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
        
        try:
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except (ValueError, TypeError):
          print(f"Invalid ROI coordinates at index {idx}: {x1}, {y1}, {x2}, {y2}")
          pass
        else:
          img_width, img_height = image.size
          if (x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and 
            x2 <= img_width and y2 <= img_height):
            image = image.crop((x1, y1, x2, y2))
          else:
            print(f"Invalid ROI bounds at index {idx}: ({x1},{y1},{x2},{y2}) for image size {img_width}x{img_height}")

      if self.transform:
        image = self.transform(image)

      return image, label
        
    except Exception as e:
      print(f"Error loading sample at index {idx}: {str(e)}")
      print(f"Row data: {row.to_dict()}")
      raise e