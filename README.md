# Multihop-framework for community detection

## Overview
  This repository provides a reference implementation of multihop framework as described in the paper:
  > Jinyu Huang and Xiaogang Jin, Multi-hop framework for community detection.

## Running the code
- The multihop-lead-eigen.py file contains the example usages of the code that implements the multihop method + the method of leading eigenvectors of the modularity matrix.
- The multihop-louvain.py file contains the example usages of the code that implements the multihop method + Louvain method.
- The multihop-spectral-clustering.py file contains the code that identifies motif-based communities of Florida Bay food web using the multihop method + regularization + spectral clustering.

## Requirements
```
python >= 3.7
numpy >= 1.18.1
networkx >=2.4
sklearn >= 0.22.1
```
