import argparse
import time
import numpy as np
from tqdm import trange
from .lib import read_sitk_as_array,norm_and_quantize_image,get_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_path",required=True)
    parser.add_argument("--output_npy",required=True)
    parser.add_argument("--method_detector",default="sift",
                        choices=["sift","kaze","akaze","fast"])
    parser.add_argument("--method_descriptor",default=None,
                        choices=["sift","kaze","akaze"])
    parser.add_argument("--retrieve_top_k",default=None,type=int)
    
    args = parser.parse_args()
    
    a = time.time()
    image = read_sitk_as_array(args.image_path)
    image = norm_and_quantize_image(image)
    
    all_features = [[],[],[]]
    for s in trange(image.shape[0]):
        kps,features = get_features(image[s],
                                    args.method_detector,
                                    args.method_descriptor,
                                    retrieve_top_k=args.retrieve_top_k)
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
    b = time.time()

    all_features.append(b-a)

    np.save(args.output_npy,np.array(all_features,dtype=object))