import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class PicaiDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        descriptors = os.path.join(self.data_dir, self.img_labels.iloc[idx, 0], self.img_labels.iloc[idx, 1])
        slices = np.load(descriptors, allow_pickle=True)[2]
        label = self.img_labels.iloc[idx, -1]
        if self.transform:
            # template if needed
            image = self.transform(slices)
        if self.target_transform:
            # template if needed
            label = self.target_transform(label)
        return slices, label