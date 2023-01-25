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

To install these packages, [`poetry`](https://python-poetry.org/) is recommended. Once poetry is installed, one has to simply run `poetry install` in the present folder and you are good to go.

## Command line interface usage (recommended)

### 1. Bag of visual words (a.k.a. bag of features)

At the beginning of each section we define a couple of terms in `code notation` to simplify references to inputs and outputs between sections. Each module has a `--help` flag that lists the available arguments and flags.

#### 1.1 Feature extraction

```
input: a (set of) MRI volume(s)
output: descriptor file
```

To extract features from a SimpleITK-readable image (`nii`, `nii.gz`, `mha`), use the `python -m mri_bovw.keypoint` module. This will produce a `descriptor file` (`.npy`) containing three lists of arrays and one float (runtime). Each list of arrays has one array for each slice and is composed in the following way (here $N_i$ is the number of detected keypoints in slice $i$ and $p$ is the size of the descriptor used to describe each keypoint):

1. An $N_i \times 2$-sized array containing the positions of the keypoints in the 2D plane
2. An $N_i$-sized array containing the responses (i.e. edge-ness) of each keypoint
3. An $N_i \times p$-sized array containg the keypoint descriptors.

#### 1.2. Clustering

##### 1.2.1. Inferring the cluster centres from descriptor files

```
input: a set of descriptor files
output: clustering model
```

Clustering can be run using `python -m mri_bovw.cluster`. Clustering can be ran on a list of input `descriptor files` (i.e. `python -m mri_bovw.cluster descriptors/*npy` if you would like to include a whole folder).

##### 1.2.2. Counting clusters in descriptor files

```
input: a set of descriptor files
input: clustering model
output: cluster counts
```

To assign to and count clusters in the `descriptor files`, `python -m mri_bovw.cluster.predict` is the module that should be used. It has a similar interface to `python -m mri_bovw.cluster` but accepts only two obligatory arguments: a list of input paths (`--input_paths`) and the path to the `clustering model` (`--model_path`). A third, optional argument (`--output_path`) should be specified to produce the `cluster counts`, which can be used in the following steps.

#### 1.3. Term-frequency * inverse term frequency (tf-idf)

```
input: cluster counts
input (optional): tf-idf file
output: tf-idf file
```

Using the output from the previous step (`descriptor files`), we can calculate the [`tf-idf`](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) files. This quantity is, essentially, the frequency of different descriptors across images weighed by the scarcity (inverse document frequency) of each. 

To calculate the `tf-idf files` for each volume, you can simply run `python -m mri_bovw.cluster.tf_idf`. To necessary arguments are: the `--input_path`, which should be the output produced in the previous step when using the `--output_path` flag, and the `--output_path`, which is the produced tf-idf vectors for each volume and the idf used to calculate it (the recommended way of loading the `tf-idf` file is using `joblib.load`, which produces a dictionary containing the tf-idf values as a dictionary (under the `features` key) and the idf values used to calculate this (under the `idf` key)). By storing the idf together with the tf-idf, we can ensure that this is transferrable to other (validation or testing datasets) - to calculate the `tf-idf file` of a `cluster count file` using the idf calculated with another dataset, we simply have to use the `--idf_source` flag pointing to the source tf-idf output.

### 2. Multiple instance learning of virtual phenotypes

To-do!