from torch.utils.data import DataLoader
from .data import PicaiDataset
from .models.transformer import Transformer, TransformerNoMHSA
import torch
from torch.utils.data import random_split
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import argparse
from torcheval.metrics.functional import binary_auroc

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
    parser.add_argument("--n_epochs", default=10, type=int,
                        help="Number of epochs")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--model_config", type=str,
                        help="Path to yaml file with model configuration parameters")
    parser.add_argument("--n_workers", default=0, type=int,
                        help="Number of concurrent processes")

    args, unknown_args = parser.parse_known_args()

    dataset = PicaiDataset(args.dataset, args.descriptors, slices=True, n_slices=16)

    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.7), len(dataset)-int(len(dataset)*0.7)],
                                              generator=torch.Generator().manual_seed(args.seed))

    # build dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    #print(train_loader)

    # calculate steps per epoch for training and validation set
    train_steps = len(train_loader.dataset) // args.batch_size
    val_steps = len(val_loader.dataset) // args.batch_size

    model = Transformer(num_classes=2).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.2e-6)

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "train_auroc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auroc": []
    }

    for e in range(0, args.n_epochs):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        # initialize the number of correct predictions in the training and validation step
        train_correct = 0
        train_auroc = 0
        val_correct = 0
        val_auroc = 0
        # loop over the training set
        for (x, y) in tqdm(train_loader):
            (x, y) = (x.float().to(args.device), y.to(args.device))
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add the loss to the total training loss so far and calculate the number of correct predictions
            total_train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            train_auroc += binary_auroc(target=y.cpu(), input=pred.argmax(1).detach().cpu())

        with torch.no_grad():
            model.eval()
            for (x, y) in val_loader:
                (x, y) = (x.float().to(args.device), y.to(args.device))
                pred = model(x)
                total_val_loss += loss_fn(pred, y)
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                val_auroc += binary_auroc(target=y.cpu(), input=pred.argmax(1).detach().cpu())
        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        # calculate the training and validation accuracy
        train_correct = train_correct / len(train_loader.dataset)
        train_auroc = train_auroc / len(train_loader.dataset)
        val_correct = val_correct / len(val_loader.dataset)
        val_auroc = val_auroc / len(val_loader.dataset)

        # update our training history
        H["train_loss"].append(avg_train_loss)
        H["train_acc"].append(train_correct)
        H["train_auroc"].append(train_auroc)
        H["val_loss"].append(avg_val_loss)
        H["val_acc"].append(val_correct)
        H["val_auroc"].append(val_auroc)
        print("[INFO] EPOCH: {}/{}".format(e + 1, args.n_epochs))
        print("Train loss: {:.6f}, Train aurocs: {:.4f}".format(
            avg_train_loss, train_auroc))
        print("Val loss: {:.6f}, Val auroc: {:.4f}\n".format(
            avg_val_loss, val_auroc))

    print('Finished Training')