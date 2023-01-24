"""
Functions for managing large collections of volume keypoints and to train
machine-learning models using these collections.
"""

__author__ = ["José Guilherme de Almeida","Nuno Rodrigues"]
__version__ = "0.1"
__maintainer__ = ["José Guilherme de Almeida","Nuno Rodrigues"]

import numpy as np
from tqdm import tqdm

from typing import List

class DescriptorGenerator:
    """Standard generator of descriptors. Allows for descriptors to be
    randomly sampled and retrieved on a per slice or per volume fashion.
    """
    def __init__(self,
                 paths:List[str],
                 batch_size:int=100,
                 output_type:str="flat",
                 seed:int=42):
        """
        :param List[str] paths: list of paths to npy files, each containing
            an object array with four elements: list of keypoint locations,
            list of responses, list of descriptions and time elapsed. For more
            information please consult the `keypoint` module.
        :param int batch_size: number of descriptors to be returned, defaults 
            to 100
        :param str output_type: type of return, either "flat" or "slice", 
            depending on whether features should be returned on a per volume or
            per slice fashion, defaults to "flat"
        :param int seed: random seed, defaults to 42
        """
        assert output_type in ["flat","slice"]
        self.paths = paths
        self.batch_size = batch_size
        self.output_type = output_type
        self.seed = seed
        
        self.rng = np.random.default_rng(self.seed)
    
    def __len__(self)->int:
        """Length method.

        :return int: length of self.paths.
        """
        return len(self.paths)
    
    def __getitem__(self,idx:int)->np.ndarray:
        """Retrieve a descriptor batch or slice by index. If the output type
        is "flat", a batch of descriptors is returned, either of random size or
        with the batch size specified in the constructor. If the output type
        is "slice", a list of random descriptor slices is returned, each with
        the batch size specified in the constructor.


        :param int idx: index of the path.
        :return np.ndarray: array containing self.batch_size descriptors
            belonging to the idx-th path.
        """
        if self.output_type == "flat":
            if self.batch_size is not None:
                return self.return_random_descriptors(
                    self.paths[idx],self.batch_size,self.rng)
            if self.batch_size is None:
                return self.return_all_descriptors(
                    self.paths[idx],self.batch_size,self.rng)
        if self.output_type == "slice":
            return self.return_random_descriptors_per_slice(
                self.paths[idx],self.batch_size,self.rng)
    
    @staticmethod
    def return_random_descriptors(
        path:str,batch_size:int=None,rng:np.random.Generator=None)->np.ndarray:
        """Returns a set of random descriptors from path.

        :param str path: path to npy file as described in __init__.
        :param int batch_size: size of the output, defaults to None
        :param np.random.Generator rng: random number generator, defaults to 
            None
        :return np.ndarray: array containing batch_size*number_of_slices 
            descriptors, where number_of_slices is the number of slices in 
            path.
        """
        if rng is not None:
            rng = np.random.default_rng()
        data_store = np.load(path,allow_pickle=True)
        output = []
        n_slices = len(data_store[2])
        p = np.ones(n_slices) / n_slices
        descriptor_per_slice = rng.multinomial(batch_size,p)
        for S,n in zip(data_store[2],descriptor_per_slice):
            output.extend(rng.choice(S,n))
        output = np.array(output,dtype=int)
        return output
    
    @staticmethod
    def return_random_descriptors_per_slice(
        path:str,batch_size:int=None,rng:np.random.Generator=None)->np.ndarray:
        """Returns a set of random descriptors for each slice in the path.

        :param str path: path to npy file as described in __init__.
        :param int batch_size: size of the output, defaults to None
        :param np.random.Generator rng: random number generator, defaults to 
            None
        :return np.ndarray: array containing batch_size descriptors.
        """
        if rng is not None:
            rng = np.random.default_rng()
        data_store = np.load(path,allow_pickle=True)
        output = []
        for S in data_store[2]:
            output.append(rng.choice(S,batch_size))
        output = np.array(output,dtype=int)
        return output

    @staticmethod
    def return_all_descriptors(path:str)->np.ndarray:
        """Returns all the descriptors belonging in a given path.

        :param str path: path to npy file as described in __init__.
        :return np.ndarray: an array with all the descriptors in path.
        """
        data_store = np.load(path,allow_pickle=True)
        output = []
        for slice in data_store[2]:
            output.extend(slice)
        output = np.array(output,dtype=int)
        return output
    
class CachedDescriptorGenerator(DescriptorGenerator):
    """Cache generator of descriptors. Allows for descriptors to be
    randomly sampled and retrieved on a per slice or per volume fashion.
    Similar to DescriptorGenerator but caches all the data to speed up
    retrieval.
    """
    def __init__(self,
                 paths:List[str],
                 batch_size:int=100,
                 output_type:str="flat",
                 seed:int=42):
        """
        :param List[str] paths: list of paths to npy files, each containing
            an object array with four elements: list of keypoint locations,
            list of responses, list of descriptions and time elapsed. For more
            information please consult the `keypoint` module.
        :param int batch_size: number of descriptors to be returned, defaults 
            to 100
        :param str output_type: type of return, either "flat" or "slice", 
            depending on whether features should be returned on a per volume or
            per slice fashion, defaults to "flat"
        :param int seed: random seed, defaults to 42
        """
        assert output_type in ["flat","slice"]
        self.paths = paths
        self.batch_size = batch_size
        self.output_type = output_type
        self.seed = seed
        
        self.n = len(paths)
        self.rng = np.random.default_rng(self.seed)
        self.generate_cache()
    
    def generate_cache(self):
        """
        Generates a cache of descriptor data from the file paths provided in 
        the constructor. Invalid file paths are ignored.
        """
        self.cache = {}
        good_paths = []
        for path in tqdm(self.paths):
            try:
                self.cache[path] = np.load(path,allow_pickle=True)[2]
                tot = sum([len(x) for x in self.cache[path]])
                if len(self.cache[path]) > 0 and tot > self.batch_size:
                    good_paths.append(path)
            except:
                pass
        self.paths = good_paths
            
    def __getitem__(self,idx:int)->np.ndarray:
        """
        Retrieve a descriptor batch or slice by index. If the output type is 
        "flat", a batch of descriptors is returned, either of random size or
        with the batch size specified in the constructor. If the output type
        is "slice", a list of random descriptor slices is returned, each with
        the batch size specified in the constructor.
        
        :param int idx: index of the path.
        :return np.ndarray: array containing self.batch_size descriptors
            belonging to the idx-th path.
        """
        if self.output_type == "flat":
            if self.batch_size is not None:
                return self.return_random_descriptors(
                    self.cache[self.paths[idx]],self.batch_size,self.rng)
            if self.batch_size is None:
                return self.return_all_descriptors(
                    self.cache[self.paths[idx]],self.batch_size,self.rng)
        if self.output_type == "slice":
            return self.return_random_descriptors_per_slice(
                self.cache[self.paths[idx]],self.batch_size,self.rng)
    
    @staticmethod
    def return_random_descriptors_per_slice(
        slice_list:List[int],
        batch_size:int=None,
        rng:np.random.Generator=None)->np.ndarray:
        """Returns a set of random descriptors for each slice in the path.

        :param List[int] slice_list: list of slice indices from which 
            descriptors will be retrieved.
        :param int batch_size: size of the output, defaults to None
        :param np.random.Generator rng: random number generator, defaults to 
            None
        :return np.ndarray: array containing batch_size descriptors.
        """
        if rng is not None:
            rng = np.random.default_rng()
        output = []
        n_slices = len(slice_list)
        is_sampled = np.array([len(x) for x in slice_list])
        is_sampled = is_sampled > (batch_size / n_slices)
        p = np.ones(n_slices) * is_sampled
        p /= p.sum()
        descriptor_per_slice = rng.multinomial(batch_size,p)
        for S,n,IS in zip(slice_list,descriptor_per_slice,is_sampled):
            output.extend(rng.choice(S,n))
        output = np.array(output,dtype=int)
        return output
    
    @staticmethod
    def return_random_descriptors_per_slice(
        slice_list:List[np.ndarray],
        batch_size:int=None,
        rng:np.random.Generator=None)->np.ndarray:
        """Returns a set of random descriptors for a list of slice descriptors.

        :param List[np.ndarray] slice_list: list of slice descriptors.
        :param int batch_size: size of the output, defaults to None
        :param np.random.Generator rng: random number generator, defaults to 
            None
        :return np.ndarray: array containing batch_size descriptors.
        """
        if rng is not None:
            rng = np.random.default_rng()
        output = []
        for S in slice_list:
            output.append(rng.choice(S,batch_size))
        output = np.array(output,dtype=int)
        return output

    @staticmethod
    def return_all_descriptors(slice_list:List[np.ndarray])->np.ndarray:
        """Returns all the descriptors in slice_list.

        :param List[np.ndarray] slice_list: list of slice descriptors.
        :return np.ndarray: array containing batch_size descriptors.
        """
        output = []
        for slice in slice_list:
            output.extend(slice)
        output = np.array(output,dtype=int)
        return output