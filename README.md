# Applying bag of visual words-like approaches to MRI images

## Models to be implemented

- [ ] Bag of visual words. Requires:
    * Clustering descriptors on training set into $k$ clusters
    * Clustering data on training set and calculate tf-idf for each per image
    * Train models on tf-idf vectors

- [ ] Multiple-instance learning approach. Requires an end-to-end model which:
    1. Given a vocabulary of $k$ terms, infer to which descriptor belongs
    2. Use the proportion of terms on each sequence to predict whether a sequence is associated with a specific class

- [ ] Transformer approach. Similar to the previous approach but has a self-attention module