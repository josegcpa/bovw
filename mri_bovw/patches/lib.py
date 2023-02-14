import math
from numpy.random import randint
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from glob import glob
import monai
import matplotlib.pyplot as plt

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

def extract_random_patches(x_train, features_per_image, W):
    '''
    x_train: images of shape 4d, for example cifar10 50Kx32x32x3 . type is ignored.
    features_per_image: average number of features per image to extract
    W: each patch will be a square W*W*D

    returns: a 2d array of shape M*(W*W*D) of same type, where M is numbers-of-images*features_per_image

    usage:
    >>> extract_random_patches(x_train[:10],100,5).shape
    (100L, 75L)
    '''
    M = x_train.shape[0] * features_per_image
    patches = np.empty(shape=(M, (W * W * x_train.shape[-1])), dtype=np.float32)

    sample = randint(0, x_train.shape[0], size=M)  # low (inclusive) to high (exclusive).
    i_pixel = randint(0, x_train.shape[1] - W + 1, size=M)
    j_pixel = randint(0, x_train.shape[2] - W + 1, size=M)
    for m in tqdm(range(M)):
        patch = x_train[sample[m],
                i_pixel[m]: W + i_pixel[m],
                j_pixel[m]: W + j_pixel[m],
                :].ravel()
        patches[m] = patch
    return patches



def enhanced_imshow(img):
    """
    images after normalization (mean,std) will be in the range of [-v,+v] , where most pixels are in [-1,+1]
    plt expect values between 0 and 1, so we check min, and add it:  [-v,+v]->[0,2v]  then divide by min+max ->[0,1]
    """
    if img.dtype not in [np.float16, np.float32, np.float64]:
        raise ValueError(
            str(img.dtype) + 'only np.float32 is supported. if you have uint8 consider using np.astype(np.float32)/255.0')
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)


def visualize_patches(patches, max_to_show=20, cols=10, D=16):
    max_to_show = min(max_to_show, patches.shape[0])
    max_rows = int(math.ceil(max_to_show / cols))

    plt.figure(figsize=(max_to_show / max_rows * 6, 4 * max_rows))  # last may be too much
    for i in range(max_to_show):
        plt.subplot(max_rows, cols, i + 1)
        W = int(math.sqrt(patches.shape[1] / D))
        # plt.imshow(patches[i].reshape((W,W,D)))
        img = patches[i].reshape((W, W, D))
        enhanced_imshow(img[:,:,-6])
    plt.show()



def main() -> None:
    import argparse

    #parser = argparse.ArgumentParser(
    #    description=main.__doc__)

    #parser.add_argument("--image_paths", required=True,
    #                    help="Paths to SITK-readable 3D volume.")

    #args = parser.parse_args()

    image_paths = glob('/home/nuno/Desktop/dataset/*/*/*_t2w.mha')[:50]

    load = monai.transforms.Compose([
        monai.transforms.LoadImage(reader='ITKReader', ensure_channel_first=True, image_only=True),
        monai.transforms.Orientation(axcodes="PLS"),
        monai.transforms.ResizeWithPadOrCrop(spatial_size=(170,170,16)),
    ])

    image_list = []
    for image_path in tqdm(image_paths):
        image_list.append(norm_and_quantize_image(load(image_path)[0]))

    patches = extract_random_patches(np.array(image_list), 100, 170)

    print(patches.shape)

    visualize_patches(patches, max_to_show=10, cols=5)