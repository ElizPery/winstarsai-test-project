import argparse
import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --- Configuration Constants ---
# Assumed location of the preprocessed data maps
IMG_DATA_CSV_DIR = os.path.dirname(__file__).replace('src/models/img_classification_model', 'src/data/img_data_csv')
# Assumed location of the raw images (used to resolve relative paths in CSV)
IMG_RAW_DIR = os.path.dirname(__file__).replace('src/models/img_classification_model', 'src/data/img_data_raw')
MODEL_ARTIFACTS_DIR = './model_artifacts/img_model'

class ImageDataset(Dataset):
    """
    Custom Dataset class for Image Classification, loading images from a base directory
    using relative paths specified in a DataFrame.
    """
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'filename' (relative path) and 'class_label'.
            root_dir (str): Base directory where all image files are located.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct full path: RAW_IMAGES_DIR / CLASSNAME / image.jpg
        relative_img_path = self.df.iloc[idx]['filename']
        img_name = os.path.join(self.root_dir, relative_img_path)
        
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {img_name}. Skipping sample.")
            return None, None # Return None and handle in DataLoader
        
        # Get label and convert to integer ID
        label = self.df.iloc[idx]['label_id']

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to handle None items from the dataset
def collate_fn(batch):
    # Filter out samples where image loading failed (where __getitem__ returned None)
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch:
        return None, None
    
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    
    return images, labels

def train_ic_model(args):
    """
    Loads data maps, initializes, and trains the Image Classification model.
    """
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    
    # 1. Load Data Maps and Prepare Labels
    try:
        train_map_path = os.path.join(IMG_DATA_CSV_DIR, 'img_data_train.csv')
        test_map_path = os.path.join(IMG_DATA_CSV_DIR, 'img_data_test.csv')
        train_df = pd.read_csv(train_map_path)
        test_df = pd.read_csv(test_map_path)
    except FileNotFoundError:
        print(f"Error: CSV map files not found. Please run 'ic_data_prep.py' first.")
        return

    # Create mapping from string label to integer ID
    CLASS_NAMES = sorted(train_df['class_label'].unique())
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Generate ID mapping for the model and future inference
    LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
    
    # Map the class labels in the DataFrame to their integer IDs
    train_df['label_id'] = train_df['class_label'].map(LABEL_TO_ID)
    test_df['label_id'] = test_df['class_label'].map(LABEL_TO_ID)

    print(f"Loaded data for {NUM_CLASSES} classes: {CLASS_NAMES}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Save the label mapping for inference script to use
    mapping_path = os.path.join(MODEL_ARTIFACTS_DIR, 'ic_label_mapping.txt')
    with open(mapping_path, 'w') as f:
        for class_name in CLASS_NAMES:
            f.write(f"{class_name}\n")
    print(f"Label mapping saved to {mapping_path}")

    # 2. Define Transformations and Datasets
    # Standard normalization values for ImageNet pre-trained models
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = ImageDataset(train_df, IMG_RAW_DIR, data_transforms['train'])
    test_dataset = ImageDataset(test_df, IMG_RAW_DIR, data_transforms['test'])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        collate_fn=collate_fn
    )
    
    # 3. Model Setup (Transfer Learning with ResNet-18)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace the final fully connected layer for the new number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            if inputs is None: continue # Skip empty batches due to failed loads

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                if inputs is None: continue

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

        test_acc = running_corrects / total_samples
        print(f'Test Acc: {test_acc:.4f}')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model_save_path = os.path.join(MODEL_ARTIFACTS_DIR, 'ic_model_weights.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with improved accuracy: {best_acc:.4f}")

    print("\nTraining complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification Model Training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    train_ic_model(args)
