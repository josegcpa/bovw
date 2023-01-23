import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans,Birch
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ..data.lib import DescriptorGenerator,CachedDescriptorGenerator
from typing import Union

def train_clustering_algorithm(
    data_generator:Union[DescriptorGenerator,CachedDescriptorGenerator],
    algo:BaseEstimator,
    n_iter:int,
    batch_image_size:int,
    tol:float,
    early_stopping:int)->BaseEstimator:
    
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
    import argparse
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_paths",nargs="+",required=True)
    parser.add_argument("--batch_image_size",type=int,default=16)
    parser.add_argument("--batch_size",type=int,default=1000)
    parser.add_argument("--n_clusters",type=int,default=10)
    parser.add_argument("--n_iter",type=int,default=1000)
    parser.add_argument("--learning_algorithm",type=str,default="kmeans",
                        choices=["kmeans","birch"])
    parser.add_argument("--early_stopping",default=-1,type=int)
    parser.add_argument("--tol",default=0.0,type=float)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output_path",type=str,default=None)
    
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
        joblib.dump(algo,filename=args.output_path)