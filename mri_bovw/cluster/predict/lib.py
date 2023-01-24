import joblib
import numpy as np

from tqdm import tqdm

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_paths",dest="input_paths",required=True,
                        nargs="+")
    parser.add_argument("--model_path",dest="model_path",required=True)

    args = parser.parse_args()
    
    algo = joblib.load(args.model_path)
    n_clusters = algo.n_clusters

    frequency = {}
    dft = np.zeros(n_clusters)

    print("Calculating frequencies...")
    with tqdm(args.input_paths) as pbar:
        for path in pbar:
            slices = np.load(path,allow_pickle=True)[2]
            cluster_f = np.zeros(n_clusters)
            for i,s in enumerate(slices):
                output = np.zeros([n_clusters],dtype=np.int32)
                if len(s.shape) == 2:
                    cluster_assignment = algo.predict(s.astype(np.float32) / 255.)
                    for u,c in zip(*np.unique(cluster_assignment,return_counts=True)):
                        output[u] += c
                    cluster_f += output
                    output = ",".join(output.astype(str))
                    output = "{},{},{}".format(path,i,output)
                    print(output)

            dft += np.nan_to_num(cluster_f/cluster_f,0,0,0)
            frequency[path] = cluster_f

    print("Calculating tf-idf...")
    with tqdm(args.input_paths) as pbar:
        for path in pbar:
            tf_idf = np.zeros(n_clusters)
            for i, term in enumerate(frequency[path]):
                tf_idf[i] = term * (len(args.input_paths) / dft[i])
            frequency[path] = tf_idf
            output = ",".join(tf_idf.astype(str))
            output = "{},{}".format(path, output)
            print(output)

