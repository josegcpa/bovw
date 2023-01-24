"""
Functions to assign clusters to descriptors.
"""

__author__ = ["José Guilherme de Almeida","Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida","Nuno Rodrigues"]

import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    """Takes a set of input_paths, reads the descriptors contained in each of
    them and assigns clusters using the model in model_path. It is expected
    that the object contained in model_path has an `n_clusters` attribute and a
    `predict` method that outputs a cluster between 0 and n_clusters.

    The sum of cluster assignments are then stored in output_path as a
    dictionary with the structure `{input_path: {slice_idx:
    cluster_assignment_count}}`, where `slice_idx` corresponds to the slice index
    and `cluster_assignment_count` corresponds to a vector counting the number
    of descriptors assigned to each cluster in the slice_idx-th slice.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=main.__doc__)

    parser.add_argument("--input_paths", dest="input_paths", required=True,
                        nargs="+",
                        help="Space-separated list of paths to descriptors.")
    parser.add_argument("--output_path", dest="output_path",
                        help="Path to cluster assignment counts.")
    parser.add_argument("--model_path", dest="model_path", required=True,
                        help="Path to clustering model.")

    args = parser.parse_args()

    algo = joblib.load(args.model_path)
    n_clusters = algo.n_clusters

    frequency = {}

    with tqdm(args.input_paths) as pbar:
        for path in pbar:
            slices = np.load(path, allow_pickle=True)[2]
            cluster_f = {}
            for i, s in enumerate(slices):
                output = np.zeros([n_clusters], dtype=np.int32)
                if len(s.shape) == 2:
                    cluster_assignment = algo.predict(
                        s.astype(np.float32) / 255.)
                    for u, c in zip(*np.unique(cluster_assignment,
                                               return_counts=True)):
                        output[u] += c
                    cluster_f[i] = output
                    output = ",".join(output.astype(str))
                    output = "{},{},{}".format(path, i, output)
            frequency[path] = cluster_f

    if args.output_path is not None:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(frequency, filename=args.output_path)