import os
import re
from glob import glob

id_pattern = "[0-9]+_[0-9]+"
descriptor_dir = "descriptors"
all_ids = glob("/home/jose_almeida/data/PI-CAI/dataset/*/*")[:200]
detector_method = "akaze"
descriptor_method = "akaze"

image_correspondence = {
    "t2":{},
    "adc":{},
    "hbv":{}}
all_descriptors = []
for k in image_correspondence:
    os.makedirs(os.path.join(descriptor_dir,k),exist_ok=True)

for pid in all_ids:
    for p in image_correspondence:
        image_path = glob(os.path.join(pid,"*{}*".format(p)))
        if len(image_path) > 0:
            image_path = image_path[0]
            b = image_path.split(os.sep)[-1]
            patient_id = re.search(id_pattern,b).group()
            descriptor_path = os.path.join(descriptor_dir,p,patient_id + ".npy")
            image_correspondence[p][patient_id] = image_path
            all_descriptors.append(descriptor_path)

rule all:
    input:
        all_descriptors

rule get_descriptors:
    input:
        lambda wc: image_correspondence[wc.seq_id][wc.patient_id]
    output:
        os.path.join(descriptor_dir,"{seq_id}","{patient_id}.npy")
    shell:
        """
        python -m mri_bovw \
            --image_path {input} \
            --method_detector {detector_method} \
            --method_descriptor {descriptor_method} \
            --output_npy {output}
        """