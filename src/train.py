import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import EmotionCNN

class SyntheticEmotionDataset(Dataset):
    """
    Generates synthetic grayscale images for demonstration and testing.
    Useful when a full dataset is not yet downloaded.
    """
    def __init__(self, num_samples=1000, num_classes=5, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform
        
        # Generate random 48x48 pixel values (0-255)
        self.images = np.random.randint(0, 256, (num_samples, 48, 48), dtype=np.uint8)
        self.labels = np.random.randint(0, num_classes, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to float tensor
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # Add channel dim: [1, 48, 48]
        
        return torch.tensor(img), torch.tensor(label, dtype=torch.long)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train EmotionCNN Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of emotion classes")
    parser.add_argument("--save_path", type=str, default="../models/emotion_cnn.pth", help="Path to save model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Load synthetic dataset for demonstration
    print("Loading synthetic datasets...")
    train_dataset = SyntheticEmotionDataset(num_samples=1000, num_classes=args.num_classes)
    val_dataset = SyntheticEmotionDataset(num_samples=200, num_classes=args.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = EmotionCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save model weights
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved successfully to {args.save_path}")

if __name__ == "__main__":
    main()
