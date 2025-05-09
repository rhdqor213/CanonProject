import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import os
import shutil
from PIL import Image
import argparse
from typing import Tuple, List
import functools

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    parser.add_argument('--model_path', type=str, default='./best_model.pth', help='path to model weights')
    parser.add_argument('--test_dir', type=str, default='./test', help='path to test dataset')
    parser.add_argument('--output_dir', type=str, default='./classified_images', help='path to output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for inference')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes')
    return parser.parse_args()

def get_device() -> torch.device:
    """Get the device to use for computation."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    """Load and prepare the model."""
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model.to(device)

def get_transforms() -> transforms.Compose:
    """Create data transformations."""
    return transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

def load_dataset(test_dir: str, transform: transforms.Compose, batch_size: int) -> Tuple[ImageFolder, DataLoader, List[str]]:
    """Load dataset and create data loader."""
    dataset = ImageFolder(root=test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    file_paths = [sample[0] for sample in dataset.samples]
    return dataset, loader, file_paths

def create_output_directories(output_dir: str, num_classes: int) -> None:
    """Create output directories for each class."""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_classes):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

def copy_image(src_path: str, class_dir: str) -> None:
    """Copy image to its predicted class directory."""
    dst_path = os.path.join(class_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)

def run_inference(model: nn.Module, loader: DataLoader, file_paths: List[str], 
                 output_dir: str, device: torch.device) -> None:
    """Run inference on the test dataset."""
    model.eval()
    
    with torch.inference_mode():
        file_idx = 0
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            
            # Save each image to its predicted class folder
            for j, pred in enumerate(preds):
                if file_idx < len(file_paths):
                    src_path = file_paths[file_idx]
                    class_dir = os.path.join(output_dir, str(pred.item()))
                    copy_image(src_path, class_dir)
                    print(f"Image {file_idx}: {src_path} → Class {pred.item()}")
                    file_idx += 1

def main() -> None:
    """Main function to run the inference pipeline."""
    args = parse_arguments()
    device = get_device()
    model = load_model(args.model_path, args.num_classes, device)
    transform = get_transforms()
    dataset, loader, file_paths = load_dataset(args.test_dir, transform, args.batch_size)
    create_output_directories(args.output_dir, args.num_classes)
    run_inference(model, loader, file_paths, args.output_dir, device)

if __name__ == "__main__":
    main()