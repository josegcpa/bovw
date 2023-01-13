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
    detector = cv2.AKAZE_create(threshold=0.002)
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

def get_features(image,descriptor):
    descriptor = descriptor.lower()
    feature_extractors = {
        "sift":get_sift_features,
        "kaze":get_kaze_features,
        "akaze":get_akaze_features}
    assert descriptor.lower() in feature_extractors
    kps,features = feature_extractors[descriptor](image)
    return kps,features