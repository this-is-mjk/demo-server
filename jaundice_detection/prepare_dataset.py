"""
Dataset Preparation Script
Splits raw images into train/val/test sets.
"""

import os
import shutil
import argparse
from pathlib import Path
import random


def prepare_dataset(
    raw_dir: str = 'raw',
    output_dir: str = 'dataset',
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
):
    """
    Split raw images into train/val/test sets.
    
    Expected raw_dir structure:
        raw/
        ├── jaundice/
        │   └── *.jpg
        └── normal/
            └── *.jpg
    
    Output structure:
        dataset/
        ├── train/
        │   ├── jaundice/
        │   └── normal/
        ├── val/
        │   ├── jaundice/
        │   └── normal/
        └── test/
            ├── jaundice/
            └── normal/
    """
    
    random.seed(seed)
    
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    if not raw_path.exists():
        raise ValueError(f"Raw directory not found: {raw_dir}")
    
    # Get classes from subdirectories
    classes = [d.name for d in raw_path.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    stats = {}
    for cls in classes:
        class_path = raw_path / cls
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
        
        # Shuffle
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        stats[cls] = {
            'total': n,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        }
        
        # Copy files
        for img in train_images:
            shutil.copy(img, output_path / 'train' / cls / img.name)
        
        for img in val_images:
            shutil.copy(img, output_path / 'val' / cls / img.name)
        
        for img in test_images:
            shutil.copy(img, output_path / 'test' / cls / img.name)
        
        print(f"{cls}: {n} images -> train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    print(f"\n✓ Dataset prepared in {output_dir}")
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('--raw-dir', type=str, default='raw')
    parser.add_argument('--output-dir', type=str, default='dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
