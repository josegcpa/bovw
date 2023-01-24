"""print("Calculating tf-idf...")
    with tqdm(args.input_paths) as pbar:
        for path in pbar:
            tf_idf = np.zeros(n_clusters)
            for i, term in enumerate(frequency[path]):
                tf_idf[i] = term * (len(args.input_paths) / dft[i])
            frequency[path] = tf_idf
            output = ",".join(tf_idf.astype(str))
            output = "{},{}".format(path, output)
            print(output)
"""