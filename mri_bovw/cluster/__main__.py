import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans,Birch
from tqdm import tqdm

from ..data.lib import CachedDescriptorGenerator

if __name__ == "__main__":
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
    
    args = parser.parse_args()
    
    print("Loading data...")
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
        print("Training MiniBatchKMeans with:")
    elif args.learning_algorithm == "birch":
        algo = Birch(n_clusters=args.n_clusters)
        print("Training Birch with:")
    params = algo.get_params()
    for k in params:
        print("\t{}={}".format(k,params[k]))
    
    counter = 0
    min_inertia = np.inf
    with tqdm(range(args.n_iter), desc='') as pbar:
        for i in range(args.n_iter):
            batch_idx = np.random.choice(
                len(data_generator),args.batch_image_size,
                replace=False)
            batch = np.concatenate([data_generator[j] for j in batch_idx])
            batch = batch.astype(np.float32) / 255. # max is 255
            algo.partial_fit(batch)
            pbar.update()
            if args.learning_algorithm == "kmeans":
                inertia = algo.inertia_ / batch.shape[0]
                pbar.set_description("step={} || inertia={}".format(i,inertia))
                if inertia < (min_inertia+args.tol):
                    min_inertia = inertia
                    counter = 0
                else:
                    counter += 1
                if args.early_stopping > 0:
                    if counter >= args.early_stopping:
                        print("Early stopping criteria reached")
                        break
            else:
                pbar.set_description("step={}".format(i))
            