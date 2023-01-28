from torch.utils.data import DataLoader
from .data import PicaiDataset
import numpy as np

def main():


    data_path = "descriptors/t2"
    annotations_file = "picai_dataset.csv"

    ds = PicaiDataset(annotations_file, data_path)

    features, label = ds[0]

    print("number of features:",len(features))
    print("feature embeding size:", features[0].shape)

    # build dataloaders
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    train_features, train_labels = next(iter(dl))
    print(f"Feature batch shape: {train_features.shape}")
    print(f"Labels batch shape: {len(train_labels)}")
