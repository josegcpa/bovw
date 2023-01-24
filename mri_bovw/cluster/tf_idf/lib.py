"""
Functions to compute tf-idf from a frequencies object.
"""

__author__ = ["José Guilherme de Almeida", "Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida", "Nuno Rodrigues"]

import joblib
import numpy as np
from pathlib import Path

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=main.__doc__)

    parser.add_argument("--input_path", dest="input_path", required=True,
                        help="Path to file with cluster count dictionary.")
    parser.add_argument("--output_path", dest="output_path",
                        help="Path to output tf-idf object.")

    args = parser.parse_args()

    frequency = joblib.load(args.input_path)
    tmp = frequency[list(frequency.keys())[0]]
    n_clusters = tmp[list(tmp.keys())[0]].shape[0]

    dft = np.zeros(n_clusters)
    freq_per_img = {}

    # calculating document frequency
    for key in frequency:
        img_cluster = np.zeros(n_clusters)
        for s in frequency[key]:
            img_cluster += frequency[key][s]
        img_cluster = img_cluster / img_cluster.sum()
        dft[img_cluster > 0] += 1
        freq_per_img[key] = img_cluster

    for key in freq_per_img:
        tf_idf = np.zeros(n_clusters)
        for i, term in enumerate(freq_per_img[key]):
            idf = len(frequency.keys()) / dft[i]
            tf_idf[i] = term * idf
        freq_per_img[key] = tf_idf

    if args.output_path is not None:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(freq_per_img, filename=args.output_path)
    else:
        for k in freq_per_img:
            print("{},{}".format(
                k,",".join([str(x) for x in freq_per_img[k]])))