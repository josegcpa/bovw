import joblib
import numpy as np

from tqdm import tqdm


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_paths", dest="input_paths", required=True,
                        nargs="+")
    parser.add_argument("--output_path", dest="output_path")
    parser.add_argument("--model_path", dest="model_path", required=True)

    args = parser.parse_args()

    algo = joblib.load(args.model_path)
    n_clusters = algo.n_clusters

    frequency = {}

    print("Calculating frequencies...")
    with tqdm(args.input_paths) as pbar:
        for path in pbar:
            slices = np.load(path, allow_pickle=True)[2]
            cluster_f = list()
            for i, s in enumerate(slices):
                output = np.zeros([n_clusters], dtype=np.int32)
                if len(s.shape) == 2:
                    cluster_assignment = algo.predict(s.astype(np.float32) / 255.)
                    for u, c in zip(*np.unique(cluster_assignment, return_counts=True)):
                        output[u] += c
                    cluster_f.append(output)
                    output = ",".join(output.astype(str))
                    output = "{},{},{}".format(path, i, output)
                    print(output)
            frequency[path] = cluster_f

    if args.output_path is not None:
        p = Path(args.output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(frequency, filename=args.output_path)