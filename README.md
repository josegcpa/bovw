# Applying bag of visual words-like approaches to MRI images

## Installation

### Requirements

Requirements are specified in `pyproject.toml` under `tool.poetry.dependencies`. For clarity, they are:

* `python>=3.8,<3.12"`
* `numpy==1.24.1`
* `opencv-contrib-python==4.7.0.68`
* `tqdm==4.64.1`
* `scikit-learn==1.2.0`
* `SimpleITK==2.2.1`
* `scipy==1.10.0`

It is likely that other requirements work, but these are the ones the package was tested on.

### Installing with poetry

To install these packages, [`poetry`](https://python-poetry.org/) is recommended. Once installed, one has to simply run `poetry install` in the present folder.

## Command line interface usage (recommended)

### Bag of visual words (a.k.a. bag of features)

#### Feature extraction

To extract features from a SimpleITK-readable image (`nii`, `nii.gz`, `mha`), use the `python -m mri_bovw.keypoint` module. Command line options are made explicit `python -m mri_bovw.keypoint --help`. This will produce a `.npy` file containing three lists of arrays and one float (runtime). Each list of arrays has one array for each slice and is composed in the following way (here $N_i$ is the number of detected keypoints in slice $i$ and $p$ is the size of the descriptor used to describe each keypoint):

1. An $N_i \times 2$-sized array containing the positions of the keypoints in the 2D plane
2. An $N_i$-sized array containing the responses (i.e. edge-ness) of each keypoint
3. An $N_i \times p$-sized array containg the keypoint descriptors.

#### Clustering

##### Inferring the cluster centres from descriptor files

Clustering can be run using `python -m mri_bovw.cluster`. Command line options are made explicit `python -m mri_bovw.cluster --help`. Clustering can be ran on a list of input files (i.e. `python -m mri_bovw.cluster descriptors/*npy` if you would like to include a whole folder).

##### Assigning clusters to descriptor files

To assign clusters to the descriptors in each file, `python -m mri_bovw.cluster.predict` is the script that should be used. It has a similar interface to `python -m mri_bovw.cluster` but accepts only two arguments: a list of input paths (`--input_paths`) and the path to the model output from the clustering (`--model_path`)

#### TF-IDF (missing)

### Multiple instance learning of virtual phenotypes

To-do!