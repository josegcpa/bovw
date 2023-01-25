import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True,
                        typr=str,
                        help="Path for csv dataset. Class must be the last column.")
    parser.add_argument("--model", required=True,
                        choices=["rf", "svm", "lr"],
                        help="Name of the machine learning model to use.")
    parser.add_argument("--seed", default=42 ,type=int,
                        help="Random seed")
    parser.add_argument("--n_folds", default=5 ,type=int,
                        help="Number of folds")
    parser.add_argument("--model_config", type=str,
                        help="Path to yaml file with model configuration parameters")
    parser.add_argument("--n_workers", default=0, type=int,
                        help="Number of concurrent processes")

    args, unknown_args = parser.parse_known_args()

    supported_models = {
        "rf": RandomForestClassifier,
        "lr": LogisticRegressionCV,
        "svm": SVC}

    model = supported_models[args.model]

    df = pd.read_csv(args.dataset)

    if args.model_config is not None:
        with open(args.model_config, 'r') as o:
            model_config = yaml.safe_load(o)
    else:
        model_config = {}

    X = df.drop(df.columns[-1])
    y = df.columns[-1].values

    cv = StratifiedKFold(args.n_folds,
                         random_state=args.seed)
    splits = cv.split(X,y)

    for i, (train_idxs, val_idxs) in enumerate(splits):
        train_X = X[train_idxs]
        train_y = y[train_idxs]
        val_X = X[val_idxs]
        val_y = y[val_idxs]

        model = model(**model_config,
                      random_state=args.seed)
        model.fit(train_X, train_y)

        preds = model.predict(val_X).tolist()
        nc = len(np.unique(val_y))
        f1 = f1_score(val_y, preds,
                      average="binary" if nc == 2 else "micro")
        print("Fold concluded\n\tF1-score={}".format(f1))
