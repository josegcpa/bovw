"""
Functions for descriptor clustering.
"""

__author__ = ["José Guilherme de Almeida","Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida","Nuno Rodrigues"]

import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans,Birch
from sklearn.base import BaseEstimator
from tqdm import tqdm
from pathlib import Path

from ..data.lib import DescriptorGenerator,CachedDescriptorGenerator
from typing import Union

def train_clustering_algorithm(
    data_generator:Union[DescriptorGenerator,CachedDescriptorGenerator],
    algo:BaseEstimator,
    n_iter:int,
    batch_image_size:int,
    tol:float,
    early_stopping:int)->BaseEstimator:
    """Trains a clustering algorithm with a partial_fit method for a maximum
    of n_iter steps 

    :param Union[DescriptorGenerator,CachedDescriptorGenerator] data_generator: _description_
    :param BaseEstimator algo: _description_
    :param int n_iter: _description_
    :param int batch_image_size: _description_
    :param float tol: _description_
    :param int early_stopping: _description_
    :return BaseEstimator: _description_
    """
    counter = 0
    min_inertia = np.inf
    with tqdm(range(n_iter), desc='') as pbar:
        for i in range(n_iter):
            batch_idx = np.random.choice(
                len(data_generator),batch_image_size,
                replace=False)
            batch = np.concatenate([data_generator[j] for j in batch_idx])
            batch = batch.astype(np.float32) / 255. # max is 255
            algo.partial_fit(batch)
            pbar.update()
            if hasattr(algo,"inertia_"):
                inertia = algo.inertia_ / batch.shape[0]
                pbar.set_description("step={} || inertia={}".format(i,inertia))
                if inertia < (min_inertia+tol):
                    min_inertia = inertia
                    counter = 0
                else:
                    counter += 1
                if early_stopping > 0:
                    if counter >= early_stopping:
                        print("Early stopping criteria reached")
                        break
            else:
                pbar.set_description("step={}".format(i))
    
    return algo

def main():
    """Reads a set of input_paths, loads them into memory and randomly samples
    a set of batch_size descriptors from a set of batch_image_size volumes and
    iteratively fits a KMeans/BIRCH model to these data.
    
    To define the aspects of this clustering process, one can set the number of 
    clusters (n_clusters), number of maximum iterations (n_iter), number of 
    early stopping steps (early_stopping) and early stopping tolerance. The 
    early stopping only works with KMeans. Early stopping checks if there has 
    been an improvement over the last early_stopping steps. If not, training
    stops.
    
    The output model is stored in output_path.
    """
    import argparse
        
    parser = argparse.ArgumentParser(
        description=main.__doc__)
    
    parser.add_argument("--input_paths",nargs="+",required=True,
                        help="Space-separated paths to descriptors.")
    parser.add_argument("--n_clusters",type=int,default=10,
                        help="Number of clusters to be fitted")
    parser.add_argument("--batch_image_size",type=int,default=16,
                        help="Number of volumes sampled per iteration.")
    parser.add_argument("--batch_size",type=int,default=1000,
                        help="Number of descriptors sampled for each volume.")
    parser.add_argument("--n_iter",type=int,default=1000,
                        help="Number of fitting iterations.")
    parser.add_argument("--learning_algorithm",type=str,default="kmeans",
                        choices=["kmeans","birch"],
                        help="Clustering algorithm.")
    parser.add_argument("--early_stopping",default=-1,type=int,
                        help="Number of early stopping checks.")
    parser.add_argument("--tol",default=0.0,type=float,
                        help="Tolerance for the early stopping.")
    parser.add_argument("--seed",type=int,default=42,
                        help="Random seed.")
    parser.add_argument("--output_path",type=str,default=None,
                        help="Output path for the trained model.")
    
    args = parser.parse_args()
    
    data_generator = CachedDescriptorGenerator(
        paths=args.input_paths,
        batch_size=args.batch_size,
        seed=args.seed)

    if args.learning_algorithm == "kmeans":
        algo = MiniBatchKMeans(
            n_clusters=args.n_clusters,
            random_state=args.seed,
            n_init='auto',
            max_iter=args.n_iter,
            batch_size=args.batch_image_size*args.batch_size)
    elif args.learning_algorithm == "birch":
        algo = Birch(n_clusters=args.n_clusters)

    algo = train_clustering_algorithm(
        data_generator=data_generator,
        algo=algo,
        n_iter=args.n_iter,
        batch_image_size=args.batch_image_size,
        tol=args.tol,
        early_stopping=args.early_stopping)

    if args.output_path is not None:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(algo,filename=args.output_path)
