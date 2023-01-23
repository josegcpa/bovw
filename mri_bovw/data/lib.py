import numpy as np
from tqdm import tqdm

from typing import List

class DescriptorGenerator:
    def __init__(self,
                 paths:List[str],
                 batch_size:int=100,
                 output_type:str="flat",
                 seed:int=42):
        assert output_type in ["flat","slice"]
        self.paths = paths
        self.batch_size = batch_size
        self.output_type = output_type
        self.seed = seed
        
        self.rng = np.random.default_rng(self.seed)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,idx:int):
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
    def return_random_descriptors(path,batch_size=None,rng=None):
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
    def return_random_descriptors_per_slice(path,batch_size=None,rng=None):
        if rng is not None:
            rng = np.random.default_rng()
        data_store = np.load(path,allow_pickle=True)
        output = []
        for S in data_store[2]:
            output.append(rng.choice(S,batch_size))
        output = np.array(output,dtype=int)
        return output

    @staticmethod
    def return_all_descriptors(path):
        data_store = np.load(path,allow_pickle=True)
        output = []
        for slice in data_store[2]:
            output.extend(slice)
        output = np.array(output,dtype=int)
        return output
    
class CachedDescriptorGenerator(DescriptorGenerator):
    def __init__(self,
                 paths:List[str],
                 batch_size:int=100,
                 output_type:str="flat",
                 seed:int=42):
        assert output_type in ["flat","slice"]
        self.paths = paths
        self.batch_size = batch_size
        self.output_type = output_type
        self.seed = seed
        
        self.n = len(paths)
        self.rng = np.random.default_rng(self.seed)
        self.generate_cache()
    
    def generate_cache(self):
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
            
    def __getitem__(self,idx:int):
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
    def return_random_descriptors(slice_list,batch_size=None,rng=None):
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
    def return_random_descriptors_per_slice(slice_list,batch_size=None,rng=None):
        if rng is not None:
            rng = np.random.default_rng()
        output = []
        for S in slice_list:
            output.append(rng.choice(S,batch_size))
        output = np.array(output,dtype=int)
        return output

    @staticmethod
    def return_all_descriptors(slice_list):
        output = []
        for slice in slice_list:
            output.extend(slice)
        output = np.array(output,dtype=int)
        return output