import time
import numpy as np
import cv2
import SimpleITK as sitk
from scipy.spatial.distance import pdist,squareform
from tqdm import trange
from typing import Tuple,List

Descriptors = Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray]]
DescriptorsAndTime = Tuple[
    List[np.ndarray],List[np.ndarray],List[np.ndarray],float]

def read_sitk_as_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def norm_and_quantize_image(image):
    m = image.min()
    M = image.max()
    image = (image - m) / (M - m)
    image = np.uint8(image * 255)
    return image

def filter_keypoints(kps):
    positions = []
    for kp in kps:
        positions.append(kp.pt)
    dists = squareform(pdist(positions))
    dists[dists == 0] = np.inf

def get_akaze_features(image):
    detector = cv2.AKAZE_create()
    keypoints,features = detector.detectAndCompute(image,None)
    return keypoints,features

def get_kaze_features(image):
    detector = cv2.KAZE_create()
    keypoints,features = detector.detectAndCompute(image,None)
    return keypoints,features

def get_sift_features(image):
    detector = cv2.SIFT_create()
    keypoints,features = detector.detectAndCompute(image,None)
    return keypoints,features

def get_sift_keypoints(image):
    detector = cv2.SIFT_create()
    keypoints = detector.detect(image,None)
    return keypoints

def get_fast_keypoints(image):
    detector = cv2.FastFeatureDetector_create()
    keypoints = detector.detect(image,None)
    return keypoints

def get_akaze_keypoints(image):
    detector = cv2.AKAZE_create()
    keypoints = detector.detect(image,None)
    return keypoints

def get_kaze_keypoints(image):
    detector = cv2.KAZE_create()
    keypoints = detector.detect(image,None)
    return keypoints

def get_features_from_detector(detector, keypoints, image):
    kps,features = detector.compute(image,keypoints)
    return kps,features

def filter_keypoints(kps,k):
    r = np.array([kp.response for kp in kps])
    s = np.argsort(-r)
    if k < s.shape[0]:
        s = s[:k]
    kps = [kps[i] for i in s]
    return kps

def get_features(image,descriptor):
    descriptor = descriptor.lower()
    feature_extractors = {
        "sift":get_sift_features,
        "kaze":get_kaze_features,
        "akaze":get_akaze_features}
    assert descriptor.lower() in feature_extractors
    kps,features = feature_extractors[descriptor](image)
    return kps,features

def get_features(image,detector,descriptor,retrieve_top_k=None):
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

def get_descriptors_from_volume(image:sitk.Image,
                                method_detector:str,
                                method_descriptor:str,
                                retrieve_top_k:int)->Descriptors:
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
                                     retrieve_top_k:int)->DescriptorsAndTime:
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
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_path",required=True)
    parser.add_argument("--output_npy",required=True)
    parser.add_argument("--method_detector",default="sift",
                        choices=["sift","kaze","akaze","fast"])
    parser.add_argument("--method_descriptor",default=None,
                        choices=["sift","kaze","akaze"])
    parser.add_argument("--retrieve_top_k",default=None,type=int)
    
    args = parser.parse_args()

    all_features = get_descriptors_from_volume_path(args.image_path,
                                                    args.method_detector,
                                                    args.method_descriptor,
                                                    args.retrieve_top_k)
    np.save(args.output_npy,np.array(all_features,dtype=object))
