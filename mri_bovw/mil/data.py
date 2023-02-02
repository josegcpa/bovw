import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class PicaiDataset(Dataset):
    def __init__(self, annotations_file, data_dir, slices=True, transform=None, target_transform=None):
        self.ids_labels, self.data, self.labels = slice_dataset(annotations_file, data_dir) if slices \
            else volume_dataset(annotations_file, data_dir)
        self.transform = transform
        self.target_transform = target_transform

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


def volume_dataset(annotations_file, descriptors):

    vol_ids, vol_desc, vol_y = [],[],[]

    ids_labels, data, labels = slice_dataset(annotations_file, descriptors)
    information = list(zip(ids_labels, data, labels))
    uids = np.unique(np.array(ids_labels))
    for uid in uids:
        id_,slices,label = map(list,zip(*[item for item in information if item[0] == uid]))
        if len(slice) != 24:
            slices = central_crop(slices)
        vol_ids.append(id_[0])
        vol_desc.append(torch.flatten(torch.tensor(slices),0,1))
        vol_y.append(label[0])

    return vol_ids, vol_desc, vol_y


def slice_dataset(annotations_file, descriptors):
    annotations_file = pd.read_csv(annotations_file)
    ids_labels = [(str(row[0]) + "_" + str(row[1]), row[-1]) for idx, row in annotations_file.iterrows()]
    ids_labels, data, labels = map(list,
                                 zip(*[(id_label[0], torch.from_numpy(feature), id_label[-1]) for id_label in ids_labels for
                                       feature in
                                       np.load(os.path.join(descriptors, id_label[0] + ".npy"), allow_pickle=True)[
                                           2]]))
    # temporary, budget 0 padding
    for i in range(len(data)):
        if data[i].shape[0] != 500:
            if data[i].shape[0] == 0:
                data[i] = torch.zeros(500, 128, dtype=torch.uint8)
            else:
                l = data[i].shape[0]
                padded = torch.zeros(500, 128, dtype=torch.uint8)
                padded[:l, :128] = data[i]
                data[i] = padded

    return ids_labels, data, labels


def central_crop(slices, target=24):
    dif = len(slices)/24
    if dif%2==0:
        for i in range(dif/2):
            slices.pop(0)
            slices.pop(-1)
    else:
        if dif == 1:
            slices.pop(-1)
        else:
            for i in range(dif - dif//2):
                slices.pop(-1)
            for i in range(dif//2):
                slices.pop(0)

    return slices