from torch.utils.data import DataLoader
from .data import PicaiDataset
from .models.transformer import Transformer
import torch
import numpy as np
from tqdm import tqdm

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

    model = Transformer(num_classes=2)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.2e-6)

    for epoch in tqdm(range(10)):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dl, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')