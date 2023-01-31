import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class PicaiDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.annotations_file = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ids_labels = [(str(row[0])+"_"+str(row[1]), row[-1]) for idx,row in self.annotations_file.iterrows()]
        self.data, self.labels = map(list, zip(*[(torch.from_numpy(feature), id_label[-1]) for id_label in self.ids_labels for feature in
                                                 np.load(os.path.join(self.data_dir, id_label[0]+".npy"), allow_pickle=True)[2]]))
        # temporary, budget 0 padding
        for i in range(len(self.data)):
            if self.data[i].shape[0] != 500:
                if self.data[i].shape[0] == 0:
                    self.data[i] =torch.zeros(500,128)
                else:
                    l = self.data[i].shape[0]
                    padded = torch.zeros(500,128)
                    padded[:l,:128] = self.data[i]
                    self.data[i] = padded

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            # template if needed
            image = self.transform(features)
        if self.target_transform:
            # template if needed
            label = self.target_transform(label)
        return features, label