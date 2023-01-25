"""
Functions to compute tf-idf from a frequencies object.
"""

__author__ = ["José Guilherme de Almeida", "Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida", "Nuno Rodrigues"]

import joblib
import numpy as np
from pathlib import Path

import pandas as pd


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=main.__doc__)

    parser.add_argument("--input_path", dest="input_path", required=True,
                        help="Path to file with cluster count dictionary.")
    parser.add_argument('--csv', dest='csv', action='store_true')
    parser.add_argument("--idf_source", dest="idf_source",default=None,
                        help="tf-idf output containing idf.")
    parser.add_argument("--output_path", dest="output_path",
                        help="Path to output tf-idf object.")

    args = parser.parse_args()

    frequency = joblib.load(args.input_path)
    tmp = frequency[list(frequency.keys())[0]]
    n_clusters = tmp[list(tmp.keys())[0]].shape[0]

    dft = np.zeros(n_clusters)
    freq_per_img = {}

    # calculating term frequency and accumulating document frequencies
    for key in frequency:
        img_cluster = np.zeros(n_clusters)
        for s in frequency[key]:
            img_cluster += frequency[key][s]
        img_cluster = img_cluster / img_cluster.sum()
        dft[img_cluster > 0] += 1
        freq_per_img[key] = img_cluster

    if args.idf_source is None:
        idf = len(frequency) / dft
    else:
        idf = joblib.load(args.idf_source)["idf"]
    # calculating tf-idf 
    for key in freq_per_img:
        tf = freq_per_img[key]
        tf_idf = tf * idf
        freq_per_img[key] = tf_idf

    output_dict = {
        "features":freq_per_img,
        "idf":idf}
    if args.output_path is not None:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(output_dict, filename=args.output_path)
        if args.csv:
            df1 = pd.DataFrame(output_dict['features'].keys(), columns=['images'])
            df2 = pd.DataFrame(output_dict['features'].values(), columns=['cluster_'+str(i) for i in range(1,n_clusters+1)])
            df = pd.concat([df1, df2], axis=1)
            df.to_csv(args.output_path+".csv", index=None)
    else:
        for k in freq_per_img:
            print("{},{}".format(
                k,",".join([str(x) for x in freq_per_img[k]])))