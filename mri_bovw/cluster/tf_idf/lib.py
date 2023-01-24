"""
Functions to compute tf-idf from a frequencies object.
"""

__author__ = ["José Guilherme de Almeida", "Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida", "Nuno Rodrigues"]

import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=main.__doc__)

    parser.add_argument("--input_path", dest="input_path", required=True,
                        help="Frequencies object.")
    parser.add_argument("--n_clusters", dest="n_clusters", required=True,
                        type=int,
                        help="Frequencies object.")
    parser.add_argument("--output_path", dest="output_path",
                        help="Path to tf-idf object.")

    args = parser.parse_args()

    frequency = joblib.load(args.input_path)
    n_clusters = args.n_clusters

    dft = np.zeros(n_clusters)
    freq_per_img = {}

    print("Calculating dft and frequencies per study...")
    with tqdm(len(frequency.keys())) as pbar:
        for key in frequency.keys():
            img_cluster = np.zeros(n_clusters)
            for s in frequency[key].keys():
                img_cluster += s
            dft += np.nan_to_num(img_cluster / img_cluster, 0, 0, 0)
            freq_per_img[key] = img_cluster

    print("Calculating tf-idf per study...")
    with tqdm(len(frequency.keys())) as pbar:
        for key in freq_per_img.keys():
            tf_idf = np.zeros(n_clusters)
            for i, term in enumerate(freq_per_img[key]):
                idf = len(frequency.keys()) / dft[i]
                if idf != 1.0:
                    tf_idf[i] = term * np.log(idf)
                else:
                    tf_idf[i] = term * idf
            freq_per_img[key] = tf_idf
            output = ",".join(tf_idf.astype(str))
            output = "{},{}".format(key, output)
            print(output)

    if args.output_path is not None:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(freq_per_img, filename=args.output_path)