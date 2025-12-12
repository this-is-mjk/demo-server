"""
Training script for Jaundice Detection Models.
Trains EfficientNet models for face and eye classification.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from model import JaundiceClassifier, create_model


def get_data_transforms():
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """Create train, val, and test dataloaders."""
    
    train_transform, val_transform = get_data_transforms()
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Calculate class weights for imbalanced data
    if use_weighted_sampling:
        class_counts = [0] * len(class_names)
        for _, label in train_dataset:
            class_counts[label] += 1
        
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [weights[label] for _, label in train_dataset]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list
) -> Dict:
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        roc_auc = 0.0
    
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def train_model(
    model_type: str = 'face',
    data_dir: str = 'dataset',
    output_dir: str = 'models',
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[str] = None
):
    """Main training function."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Training {model_type} model on {device}")
    
    # Create model
    model = create_model(model_type=model_type, pretrained=True)
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir, batch_size=batch_size
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Stage 1: Train classifier only
    model.freeze_backbone()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    
    best_val_acc = 0.0
    output_path = os.path.join(output_dir, model_type)
    os.makedirs(output_path, exist_ok=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n--- Stage 1: Training classifier head ---")
    for epoch in range(min(5, epochs)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model.model_name,
                'class_names': class_names,
                'val_acc': val_acc
            }, os.path.join(output_path, 'best_model.pth'))
    
    # Stage 2: Fine-tune entire model
    print("\n--- Stage 2: Fine-tuning entire model ---")
    model.unfreeze_backbone()
    optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    for epoch in range(5, epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model.model_name,
                'class_names': class_names,
                'val_acc': val_acc
            }, os.path.join(output_path, 'best_model.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'class_names': class_names,
        'val_acc': val_acc
    }, os.path.join(output_path, 'final_model.pth'))
    
    # Evaluate on test set
    checkpoint = torch.load(os.path.join(output_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n--- Test Set Evaluation ---")
    test_results = evaluate_model(model, test_loader, device, class_names)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"ROC AUC: {test_results['roc_auc']:.4f}")
    print(test_results['classification_report'])
    
    # Save results
    with open(os.path.join(output_path, 'test_results.json'), 'w') as f:
        json.dump({
            'accuracy': test_results['accuracy'],
            'roc_auc': test_results['roc_auc'],
            'confusion_matrix': test_results['confusion_matrix']
        }, f, indent=2)
    
    # Save config
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump({
            'model_type': model_type,
            'model_name': model.model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'best_val_acc': best_val_acc
        }, f, indent=2)
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    
    plt.savefig(os.path.join(output_path, 'training_results.png'))
    plt.close()
    
    print(f"\n✓ Model saved to {output_path}")
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train jaundice detection model')
    parser.add_argument('--model-type', type=str, default='face', choices=['face', 'eyes'])
    parser.add_argument('--data-dir', type=str, default='dataset')
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    
    train_model(
        model_type=args.model_type,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
