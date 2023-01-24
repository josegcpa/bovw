"""
Reads an image specified with image_path and uses method_detector to 
detect keypoints and method_descriptor to characterize them, saving 
the output in output_npy.

The output - an object array containing the keypoints location, their
response, their descriptions and a float with time elapsed - is saved 
in output_npy. 

To retrieve only the top-k descriptors according to their response 
value, retrieve_top_k can be used to specify k.

Usage:
    python -m mri_bovw.keypoint \
        --image_path IMAGE_PATH \
        --output_npy OUTPUT_NPY \
        [--method_detector {sift,kaze,akaze,fast}] \
        [--method_descriptor {sift,kaze,akaze}] \
        [--retrieve_top_k RETRIEVE_TOP_K]

options:
  -h, --help            show this help message and exit
  --image_path IMAGE_PATH
                        Path to SITK-readable 3D volume.
  --output_npy OUTPUT_NPY
                        Path to npy output.
  --method_detector {sift,kaze,akaze,fast}
                        Method used to detect keypoints in each slice.
  --method_descriptor {sift,kaze,akaze}
                        Method used to characterise keypoints.
  --retrieve_top_k RETRIEVE_TOP_K
                        Calculates features only for the top-k keypoints according to their response.
"""

__author__ = ["José Guilherme de Almeida","Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida","Nuno Rodrigues"]

from .lib import main

if __name__ == "__main__":
    main()
