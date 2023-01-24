"""
Functions for keypoint detection and description from SITK-readable volumes.
"""

__author__ = ["José Guilherme de Almeida","Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida","Nuno Rodrigues"]

import time
import numpy as np
import cv2
import SimpleITK as sitk
from tqdm import trange
from typing import Tuple,List,Union

Descriptions = Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray]]
DescriptionsAndTime = Tuple[
    List[np.ndarray],List[np.ndarray],List[np.ndarray],float]
Keypoints = List[cv2.KeyPoint]
Descriptor = Union[cv2.KAZE,cv2.AKAZE,cv2.SIFT]
Detector = Union[cv2.KAZE,cv2.AKAZE,cv2.SIFT,cv2.FastFeatureDetector]

def read_sitk_as_array(path:str)->np.ndarray:
    """Reads an SimpleITK-readable image from a path and converts it to a numpy
    array.

    :param str path: path to SimpleITK-readable image (nii, nii.gz, mha, etc.)
    :return np.ndarray: numpy array obtained from the image in path.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def norm_and_quantize_image(image:np.ndarray)->np.ndarray:
    """Normalizes and quantizes and image (all pixel values are normalized to
    be np.uint8 between 0 and 255).

    :param np.ndarray image: array corresponding to an image.
    :return np.ndarray: normalized and quantized image.
    """
    m = image.min()
    M = image.max()
    image = (image - m) / (M - m)
    image = np.uint8(image * 255)
    return image

def get_akaze_features(image:np.ndarray)->Tuple[Keypoints,Descriptions]:
    """Gets keypoints and descriptors using an AKAZE feature detector and
    descriptor.

    :param np.ndarray image: 2d np.uint8 image.
    :return Tuple[Keypoints,Descriptions]: a tuple containing the keypoints and
        features obtained from image.
    """
    detector = cv2.AKAZE_create()
    keypoints,features = detector.detectAndCompute(image,None)
    return keypoints,features

def get_kaze_features(image:np.ndarray)->Tuple[Keypoints,Descriptions]:
    """Gets keypoints and descriptors using an KAZE feature detector and
    descriptor.

    :param np.ndarray image: 2d np.uint8 image.
    :return Tuple[Keypoints,Descriptions]: a tuple containing the keypoints and
        features obtained from image.
    """
    detector = cv2.KAZE_create()
    keypoints,features = detector.detectAndCompute(image,None)
    return keypoints,features

def get_sift_features(image:np.ndarray)->Tuple[Keypoints,Descriptions]:
    """Gets keypoints and descriptors using an SIFT feature detector and
    descriptor.

    :param np.ndarray image: 2d np.uint8 image.
    :return Tuple[Keypoints,Descriptions]: a tuple containing the keypoints and
        features obtained from image.
    """
    detector = cv2.SIFT_create()
    keypoints,features = detector.detectAndCompute(image,None)
    return keypoints,features

def get_sift_keypoints(image:np.ndarray)->Keypoints:
    """Gets keypoints a SIFT feature detector.

    :param np.ndarray image: 2d np.uint8 image.
    :return Keypoints: a tuple containing the keypoints in image.
    """
    detector = cv2.SIFT_create()
    keypoints = detector.detect(image,None)
    return keypoints

def get_fast_keypoints(image:np.ndarray)->Keypoints:
    """Gets keypoints a FAST feature detector.

    :param np.ndarray image: 2d np.uint8 image.
    :return Keypoints: a tuple containing the keypoints in image.
    """
    detector = cv2.FastFeatureDetector_create()
    keypoints = detector.detect(image,None)
    return keypoints

def get_akaze_keypoints(image:np.ndarray)->Keypoints:
    """Gets keypoints an AKAZE feature detector.

    :param np.ndarray image: 2d np.uint8 image.
    :return Keypoints: a tuple containing the keypoints in image.
    """
    detector = cv2.AKAZE_create()
    keypoints = detector.detect(image,None)
    return keypoints

def get_kaze_keypoints(image:np.ndarray)->Keypoints:
    """Gets keypoints a KAZE feature detector.

    :param np.ndarray image: 2d np.uint8 image.
    :return Keypoints: a tuple containing the keypoints in image.
    """
    detector = cv2.KAZE_create()
    keypoints = detector.detect(image,None)
    return keypoints

def get_features_from_detector(descriptor:Descriptor,
                               keypoints:Keypoints,
                               image:np.ndarray)->Tuple[Keypoints,Descriptions]:
    """Uses an arbitrary keypoint descriptor to describe the keypoints obtained
    for a given image.

    :param Descriptor descriptor: feature detector with a `compute` method.
    :param Keypoints keypoints: list of keypoints.
    :param np.ndarray image: 2d uint8 image.
    :return Tuple[Keypoints,Descriptions]: a tuple containing the keypoints and
        features obtained from image.
    """
    kps,features = descriptor.compute(image,keypoints)
    return kps,features

def filter_keypoints(kps:Keypoints,k:int)->Keypoints:
    """Filters keypoints using the response values, keeping the top-k 
    keypoints.

    :param Keypoints kps: list of keypoints.
    :param int k: number of keypoints to return.
    :return Keypoints: list of the k top keypoints according to their response
        value.
    """
    r = np.array([kp.response for kp in kps])
    s = np.argsort(-r)
    if k < s.shape[0]:
        s = s[:k]
    kps = [kps[i] for i in s]
    return kps

def get_features(image:np.ndarray,
                 descriptor:Union[str,Detector])->Tuple[Keypoints,Descriptions]:
    """Sets descriptor (either a Detector or "sift", "kaze", "akaze") and uses 
    this to detect keypoints and describe them for image.

    :param np.ndarray image: 2d uint8 image.
    :param Detector descriptor: detector used to detect and describe keypoints.
    :return Tuple[Keypoints,Descriptions]: a tuple containing the keypoints and
        features obtained from image.
    """
    if isinstance(descriptor,str):
        descriptor = descriptor.lower()
        feature_extractors = {
            "sift":get_sift_features,
            "kaze":get_kaze_features,
            "akaze":get_akaze_features}
        assert descriptor.lower() in feature_extractors
    kps,features = feature_extractors[descriptor](image)
    return kps,features

def get_features(image:np.ndarray,
                 detector:Detector,
                 descriptor:Descriptor,
                 retrieve_top_k:int=None)->Tuple[Keypoints,Descriptions]:
    """Uses a detector ("fast" ,"sift", "kaze", "akaze") and a descriptor 
    ("sift", "kaze", "akaze") to detect and describe keypoints in an image.

    :param np.ndarray image: 2d uint8 image.
    :param Detector descriptor: detector used to detect and describe keypoints.
    :param Descriptor descriptor: feature detector with a `compute` method.
    :param int retrieve_top_k: retrieves the top-k descriptors with the highest
        response.
    :return Tuple[Keypoints,Descriptions]: a tuple containing the keypoints and
        features obtained from image.
    """
    detector = detector.lower()
    if descriptor is None:
        if detector == "fast":
            descriptor = "sift"
        else:
            descriptor = detector
    descriptor = descriptor.lower()
    image_detectors = {
        "sift":get_sift_keypoints,
        "fast":get_fast_keypoints,
        "kaze":get_kaze_keypoints,
        "akaze":get_akaze_keypoints}
    feature_extractors = {
        "sift":cv2.SIFT_create(),
        "kaze":cv2.KAZE_create(),
        "akaze":cv2.AKAZE_create()}
    assert detector.lower() in image_detectors
    assert descriptor.lower() in feature_extractors
    kps = image_detectors[detector](image)
    if retrieve_top_k is not None:
        kps = filter_keypoints(kps,retrieve_top_k)
    if len(kps) > 0:
        kps,features = get_features_from_detector(
            feature_extractors[descriptor],kps,image)
    else:
        features = []
    return kps,features

def get_descriptors_from_volume(image:np.ndarray,
                                method_detector:str,
                                method_descriptor:str,
                                retrieve_top_k:int)->List[Descriptions]:
    """Uses a detector ("fast" ,"sift", "kaze", "akaze") and a descriptor 
    ("sift", "kaze", "akaze") to detect and describe keypoints in a volume.

    :param np.ndarray image: 3d numpy array (assumes first dimension is the
        slice dimension).
    :param Detector descriptor: detector used to detect and describe keypoints.
    :param Descriptor descriptor: feature detector with a `compute` method.
    :param int retrieve_top_k: retrieves the top-k descriptors with the highest
        response.
    :return List[Descriptions]: list of features (one array per slice) obtained
        from image.
    """
    image = norm_and_quantize_image(image)
    
    all_features = [[],[],[]]
    for s in trange(image.shape[0]):
        kps,features = get_features(image[s],
                                    method_detector,
                                    method_descriptor,
                                    retrieve_top_k=retrieve_top_k)
        if len(features) > 0:
            positions = np.array([list(kp.pt) for kp in kps])
            responses = np.array([kp.response for kp in kps])
            # sort everything by response
            response_sort = np.argsort(-responses)
            features = features[response_sort]
            responses = responses[response_sort]
            positions = positions[response_sort]
        else:
            features = np.array([])
            responses = np.array([])
            positions = np.array([])
        
        # store in dict
        all_features[0].append(positions)
        all_features[1].append(responses)
        all_features[2].append(features.astype(np.uint8))
    return all_features

def get_descriptors_from_volume_path(image_path:str,
                                     method_detector:str,
                                     method_descriptor:str,
                                     retrieve_top_k:int)->DescriptionsAndTime:
    """Uses a detector ("fast" ,"sift", "kaze", "akaze") and a descriptor 
    ("sift", "kaze", "akaze") to detect and describe keypoints in the 
    SITK-readable image in image_path.

    :param str image_path: path to SITK-readable image.
    :param Detector descriptor: detector used to detect and describe keypoints.
    :param Descriptor descriptor: feature detector with a `compute` method.
    :param int retrieve_top_k: retrieves the top-k descriptors with the highest
        response.
    :return DescriptionsAndTime: a tuple containing the keypoints' location,
        their response, their descriptions and a float with time elapsed.
    """
    a = time.time()
    image = read_sitk_as_array(image_path)
    all_features = get_descriptors_from_volume(image,
                                               method_detector,
                                               method_descriptor,
                                               retrieve_top_k)
    b = time.time()
    all_features.append(b-a)
    return all_features

def main()->None:
    """Wrapper around the CLI function. Reads an image specified with 
    image_path and uses method_detector to detect keypoints and 
    method_descriptor to characterize them, saving the output in output_npy.
    
    The output - an object array containing the keypoints location, their
    response, their descriptions and a float with time elapsed - is saved in
    output_npy. 
    
    To retrieve only the top-k descriptors according to their response value,
    retrieve_top_k can be used to specify k.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description=main.__doc__)
    
    parser.add_argument("--image_path",required=True,
                        help="Path to SITK-readable 3D volume.")
    parser.add_argument("--output_npy",required=True,
                        help="Path to npy output.")
    parser.add_argument("--method_detector",default="sift",
                        choices=["sift","kaze","akaze","fast"],
                        help="Method used to detect keypoints in each slice.")
    parser.add_argument("--method_descriptor",default=None,
                        choices=["sift","kaze","akaze"],
                        help="Method used to characterise keypoints.")
    parser.add_argument("--retrieve_top_k",default=None,type=int,
                        help="Calculates features only for the top-k \
                            keypoints according to their response.")
    
    args = parser.parse_args()

    all_features = get_descriptors_from_volume_path(args.image_path,
                                                    args.method_detector,
                                                    args.method_descriptor,
                                                    args.retrieve_top_k)
    np.save(args.output_npy,np.array(all_features,dtype=object))
