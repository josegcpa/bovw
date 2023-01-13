import numpy as np
import cv2
import SimpleITK as sitk
from scipy.spatial.distance import pdist,squareform
from scipy.cluster import hierarchy

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
    features = detector.compute(image,keypoints)
    return features

def get_features(image,descriptor):
    descriptor = descriptor.lower()
    feature_extractors = {
        "sift":get_sift_features,
        "kaze":get_kaze_features,
        "akaze":get_akaze_features}
    assert descriptor.lower() in feature_extractors
    kps,features = feature_extractors[descriptor](image)
    return kps,features

def get_features(image, detector, descriptor):
    detector = detector.lower()
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
    features =  get_features_from_detector(feature_extractors[descriptor], kps, image)
    return kps,features