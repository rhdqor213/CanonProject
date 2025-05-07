import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import random
from typing import Dict, Tuple, List, Any, Optional

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Image Classification with ResNet18')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=256, help='batch size for validation')
    parser.add_argument('--epochs', type=int, default=4, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data_dir', type=str, default='./train_data', help='path to dataset')
    parser.add_argument('--val_split', type=float, default=0.2, help='validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    return parser.parse_args()

def get_transforms() -> transforms.Compose:
    """Create data transformations."""
    return transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

def load_and_split_data(data_dir: str, transform: transforms.Compose, val_split: float, seed: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """Load dataset and split into train and validation sets."""
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    return dataset, train_dataset, val_dataset

def create_data_loaders(train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, batch_size: int, val_batch_size: int, seed: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create data loaders for training and validation."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

def initialize_model(num_classes: int, device: torch.device) -> nn.Module:
    """Initialize and prepare the model."""
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def train_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, total_epochs: int) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for inputs, labels in train_loop:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        train_loop.set_postfix(loss=loss.item())
    
    return running_loss / len(train_loader.dataset)

def validate(model: nn.Module, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, 
             device: torch.device, epoch: int, total_epochs: int) -> Tuple[float, float, float, List[int], List[int]]:
    """Validate the model and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0.0
    
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]")
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return val_loss, val_acc, val_f1, all_preds, all_labels

def log_metrics(writer: SummaryWriter, epoch: int, train_loss: float, val_loss: float, val_acc: float, val_f1: float) -> None:
    """Log metrics to tensorboard."""
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy/validation', val_acc, epoch)
    writer.add_scalar('F1-score/validation', val_f1, epoch)

def save_model_if_best(model: nn.Module, val_f1: float, best_f1: float) -> float:
    """Save model if it has the best F1 score so far."""
    if val_f1 > best_f1:
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved with F1 score: {val_f1:.4f}")
        return val_f1
    return best_f1

def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True # this will slow down the training
    # torch.backends.cudnn.benchmark = False # this will slow down the training

def main() -> None:
    """Main function to run the training pipeline."""
    args = parse_arguments()
    set_seed(args.seed)
    writer = SummaryWriter()
    transform = get_transforms()
    
    # Load and prepare data
    dataset, train_dataset, val_dataset = load_and_split_data(args.data_dir, transform, args.val_split, args.seed)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, args.batch_size, args.val_batch_size, args.seed)
    
    # Setup model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(len(dataset.classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    best_f1 = 0.0
    for epoch in range(args.epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        
        # Validation phase
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device, epoch, args.epochs)
        
        # Log and save
        log_metrics(writer, epoch, train_loss, val_loss, val_acc, val_f1)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        best_f1 = save_model_if_best(model, val_f1, best_f1)

    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()