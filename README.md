# Structured Random Rotation-based Compression (srrcomp)

`srrcomp` offers compression techniques grounded in structured random rotation, with strong theoretical guarantees, as detailed in the following publications:

- Shay Vargaftik, Ran Ben-Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben-Itzhak, and Michael Mitzenmacher. ["DRIVE: One-bit Distributed Mean Estimation."](https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html) Advances in Neural Information Processing Systems 34 (2021): 362-377.

- Shay Vargaftik, Ran Ben Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben Itzhak, and Michael Mitzenmacher. ["EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning."](https://proceedings.mlr.press/v162/vargaftik22a.html) In International Conference on Machine Learning, pp. 21984-22014. PMLR, 2022.

Also, see the following blog for a high-level overview: 
["Pushing the Limits of Network Efficiency for Federated Machine Learning"](https://octo.vmware.com/pushing-the-limits-of-network-efficiency-for-federated-learning/)

In particular, srrcomp can be used for: 

- Fast and efficient lossy compression.
- Unbiased estimates.
- Distributed mean estimation.
- Compressing gradient updates in distributed and federated learning.

The implementation is torch-based and thus supports CPU and GPU.

The compression and decompression are being executed over the device in which the corresponding vector is stored in.

`srrcomp` currently contains the implementation of [EDEN](https://proceedings.mlr.press/v162/vargaftik22a.html).

## CUDA acceleration

srrcomp offers some functions in CUDA for faster execution (up to an order of magnitude). This acceleration requires local compilation with `nvcc`/`torch`/`python` compatible versions. 

The 'gpuacctype' argument (GPU acceleration type) is set by default to 'cuda', and can be set to 'torch' to use the torch-based implementation. 

The torch-based implementation is used, in case the CUDA acceleration is unavailable (e.g., when the verctor in CPU, or local CUDA compilation has not been done). 

## Pre-requisites

`torch` 

`numpy`

[Optional] `nvcc` for compiling the aforementioned CUDA functions for faster execution


## Installation

### Install from pip

Linux: `$ pip install srrcomp`

Windows: `$ pip install srrcomp --extra-index-url https://download.pytorch.org/whl/ --no-cache`

If the message *"Faster CUDA implementation for Hadamard and bit packing is not available. Using torch implementation instead."* appears when importing srrcomp on a GPU machine, try installing `srrcomp` from source.

### Install from source

For Windows and Ubuntu versions earlier than 22.04, download source from [the official repository](https://github.com/shayvar/structured-random-rotation-based-compression) and run `$ python setup.py install`

For Ubuntu 22.04 use build and pip and other standards-based tools to build and install from source.

## Testing

### Basic

Execute from \tests folder:
`$ python basic_test.py`

`dim`, `bits`, and `seed` variables can be modified within the script.


### Distributed Mean Estimation (DME)

Execute from \tests folder:
`$ python dme_test.py`

Use `$ python dme_test.py -h` to get the test options 

## The team

Shay Vargaftik (VMware Research), shayv@vmware.com

Yaniv Ben-Itzhak (VMware Research), ybenitzhak@vmware.com
