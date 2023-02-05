import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class PicaiDataset(Dataset):
    def __init__(self, annotations_file, data_dir, slices=True, n_slices=24, max_kp=100, transform=None, target_transform=None):
        self.ids_labels, self.data, self.labels = slice_dataset(annotations_file, data_dir, max_kp) if slices \
            else volume_dataset(annotations_file, data_dir, n_slices, max_kp)
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


def volume_dataset(annotations_file, descriptors, n_slices=24, max_kp=100):

    vol_ids, vol_desc, vol_y = [],[],[]

    ids_labels, data, labels = slice_dataset(annotations_file, descriptors, max_kp)
    information = list(zip(ids_labels, data, labels))
    uids = np.unique(np.array(ids_labels))
    for uid in uids:
        id_,slices,label = map(list,zip(*[item for item in information if item[0] == uid]))
        if len(slices) != n_slices:
            slices = central_crop(slices, n_slices)
        vol_ids.append(id_[0])
        vol_desc.append(torch.flatten(torch.stack(slices),0,1).numpy())
        vol_y.append(label[0])

    return vol_ids, vol_desc, vol_y


def slice_dataset(annotations_file, descriptors, max_kp=100):
    annotations_file = pd.read_csv(annotations_file)
    ids_labels = [(str(row[0]) + "_" + str(row[1]), row[-1]) for idx, row in annotations_file.iterrows()]
    ids_labels, data, labels = map(list,
                                 zip(*[(id_label[0], torch.from_numpy(feature), id_label[-1]) for id_label in ids_labels for
                                       feature in
                                       np.load(os.path.join(descriptors, id_label[0] + ".npy"), allow_pickle=True)[
                                           2]]))

    n_descriptors = data[0].shape[1]
    # temporary, budget 0 padding
    # zero array needs to be crated each iteration because of object reference
    for i in range(len(data)):
        if data[i].shape[0] != max_kp:
            if data[i].shape[0] == 0:
                data[i] = torch.zeros(max_kp, n_descriptors, dtype=torch.uint8)
            else:
                l = data[i].shape[0]
                padded = torch.zeros(max_kp, n_descriptors, dtype=torch.uint8)
                padded[:l, :n_descriptors] = data[i]
                data[i] = padded

    return ids_labels, data, labels


def central_crop(slices, target=24):
    dif = len(slices) - target
    if dif%2==0:
        for i in range(dif//2):
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
