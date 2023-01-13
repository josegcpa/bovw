import argparse
import time
import json
import numpy as np
from tqdm import trange
from .lib import read_sitk_as_array,norm_and_quantize_image,get_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_path",required=True)
    parser.add_argument("--method",default="sift",
                        choices=["sift","kaze","akaze"])
    
    args = parser.parse_args()
    
    a = time.time()
    image = read_sitk_as_array(args.image_path)
    image = norm_and_quantize_image(image)
    
    all_features = {"slices":[],"positions":[],"responses":[]}
    for s in trange(image.shape[0]):
        kps,features = get_features(image[s],"sift")
        positions = np.array([list(kp.pt) for kp in kps])
        responses = np.array([kp.response for kp in kps])
        # sort everything by response
        response_sort = np.argsort(-responses)
        features = features[response_sort]
        responses = responses[response_sort]
        positions = positions[response_sort]
        # store in dict
        all_features["positions"].append(positions.tolist())
        all_features["responses"].append(responses.tolist())
        all_features["slices"].append(features.tolist())
    b = time.time()

    all_features["time_elapsed"] = b-a
        
    print(json.dumps(all_features))