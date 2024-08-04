import custom_dataset
import torch
from torch.utils.data import DataLoader, random_split

def collate_fn(batch):
    images, annotations = zip(*batch)
    return torch.stack(images), annotations

def get_dataloaders(dir, batch_size, transforms):
    dataset = custom_dataset.CustomDataset(root_dir=dir, transform=transforms)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return DataLoader(train_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      collate_fn=collate_fn),\
           DataLoader(test_dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      collate_fn=collate_fn)
