"""
Takes a set of input_paths, reads the descriptors contained in each of 
them and assigns clusters using the model in model_path. It is expected 
that the object contained in model_path has an `n_clusters` attribute and a
`predict` method that outputs a cluster between 0 and n_clusters.

The sum of cluster assignments are then stored in output_path as a 
dictionary with the structure `{input_path: {slice_idx: 
cluster_assignment_count}}`, where `slice_idx` corresponds to the slice index
and `cluster_assignment_count` corresponds to a vector counting the number
of descriptors assigned to each cluster in the slice_idx-th slice.
"""

__author__ = ["José Guilherme de Almeida","Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida","Nuno Rodrigues"]

from .lib import main

if __name__ == "__main__":
    main()