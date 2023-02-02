from torch.utils.data import DataLoader
from .data import PicaiDataset
from .models.transformer import Transformer
import torch
import numpy as np
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True,
                        type=str,
                        help="Path for csv dataset. Class must be the last column.")
    parser.add_argument("--descriptors", required=True,
                        help="Path to the folder containing the descriptors.")
    parser.add_argument("--device", default="cpu", type=str,
                        help="Device to run the model on.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed")
    parser.add_argument("--n_folds", default=5, type=int,
                        help="Number of folds")
    parser.add_argument("--model_config", type=str,
                        help="Path to yaml file with model configuration parameters")
    parser.add_argument("--n_workers", default=0, type=int,
                        help="Number of concurrent processes")

    args, unknown_args = parser.parse_known_args()

    ds = PicaiDataset(args.dataset, args.descriptors, slices=True)

    # build dataloaders
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=1)

    # calculate steps per epoch for training and validation set
    train_steps = len(dl.dataset) // 64

    model = Transformer(num_classes=2).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.2e-6)

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
    }

    for e in range(0, 5):
        model.train()
        total_train_loss = 0
        # initialize the number of correct predictions in the training and validation step
        train_correct = 0
        # loop over the training set
        for (x, y) in tqdm(dl):
            (x, y) = (x.float().to(args.device), y.to(args.device))
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far and calculate the number of correct predictions
            total_train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).type(
                torch.float).sum().item()
        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        # calculate the training and validation accuracy
        train_correct = train_correct / len(dl.dataset)
        # update our training history
        H["train_loss"].append(avg_train_loss)
        H["train_acc"].append(train_correct)
        print("[INFO] EPOCH: {}/{}".format(e + 1, 5))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            avg_train_loss, train_correct))

    print('Finished Training')