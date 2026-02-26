"""
03_augmentation.py - Physically-plausible data augmentation for ML.

Demonstrates how to integrate PhysicalAugmentation into a training pipeline.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from easyppisp.tasks import PhysicalAugmentation

class DummyLinearDataset(Dataset):
    def __len__(self): return 10
    def __getitem__(self, idx): return torch.rand(128, 128, 3)

def main():
    dataset = DummyLinearDataset()
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    
    # Initialize augmentation
    aug = PhysicalAugmentation(
        exposure_range=(-1.5, 1.5),
        vignetting_range=(0.0, 0.2),
        color_jitter=0.03,
        crf_jitter=0.05
    )
    
    print("Running batch augmentation...")
    for batch in loader:
        # Augmentation is thread-safe and can run inside workers
        # or as a post-processing step on the batch.
        augmented = aug(batch)
        print(f"Batch shape: {augmented.shape} | Max value: {augmented.max():.3f}")
        break

if __name__ == "__main__":
    main()
