"""
Reads a set of input_paths, loads them into memory and randomly samples
a set of batch_size descriptors from a set of batch_image_size volumes and
iteratively fits a KMeans/BIRCH model to these data.

To define the aspects of this clustering process, one can set the number of 
clusters (n_clusters), number of maximum iterations (n_iter), number of 
early stopping steps (early_stopping) and early stopping tolerance. The 
early stopping only works with KMeans. 

The output model is stored in output_path.

Usage:
    python -m mri_bovw.cluster \
        --input_paths INPUT_PATHS [INPUT_PATHS ...]\
        [--n_clusters N_CLUSTERS] \
        [--batch_image_size BATCH_IMAGE_SIZE] \
        [--batch_size BATCH_SIZE] \
        [--n_iter N_ITER] \
        [--learning_algorithm {kmeans,birch}] \
        [--early_stopping EARLY_STOPPING] \
        [--tol TOL] [--seed SEED] \
        [--output_path OUTPUT_PATH]

options:
  -h, --help            show this help message and exit
  --input_paths INPUT_PATHS [INPUT_PATHS ...]
                        Space-separated paths to descriptors.
  --n_clusters N_CLUSTERS
                        Number of clusters to be fitted
  --batch_image_size BATCH_IMAGE_SIZE
                        Number of volumes sampled per iteration.
  --batch_size BATCH_SIZE
                        Number of descriptors sampled for each volume.
  --n_iter N_ITER       Number of fitting iterations.
  --learning_algorithm {kmeans,birch}
                        Clustering algorithm.
  --early_stopping EARLY_STOPPING
                        Number of early stopping checks.
  --tol TOL             Tolerance for the early stopping.
  --seed SEED           Random seed.
  --output_path OUTPUT_PATH
                        Output path for the trained model.
"""

__author__ = ["José Guilherme de Almeida","Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida","Nuno Rodrigues"]


from .lib import main

if __name__ == "__main__":
    main()