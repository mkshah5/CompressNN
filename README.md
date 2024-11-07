# CompressNN
A PyTorch-based Python package for easily integrating compression into PyTorch neural networks

## Requirements
- Python >= 3.9
- PyTorch >= 2.2.0
- CUDA Version 12.1 or higher

## Installing
First, clone this repository:

```
git clone https://github.com/mkshah5/CompressNN.git
```

Next, install cuSZp as follows:

```
cd ./CompressNN/external_compressors
pip install .
```

Lastly, install CompressNN:

```
cd ..           # go back to CompressNN root directory
pip install .   # Install CompressNN
```

## Usage
CompressNN wraps your PyTorch networks (`torch.nn.Module` objects) in a `CompressNNModel` (also a `torch.nn.Module`). `CompressNNModel` allows you to specify a compressor to compress activation data as well as parameters for that compressor. Currently, CompressNN supports the following compressors:
- **cuSZp**: A lossy floating point compressor for GPU (https://github.com/szcompressor/cuSZp)
- **CPUCompressor**: Transfers activations to and from host memory to save device memory

Check out `compressnn/compressors/cuszpcompress.py` and `compressnn/compressors/cpucompress.py` for source code and arguments required. Given variable `model` is a `torch.nn.Module`, do the following:

```
model = CompressNNModel(model,<compressor>,<error_bound_mode>,<error_bound>,<compress_check>,<free_space>,<get_debug>)
```

`model` should now be able to run like any other model, with the integration of compression.

## Example
Run the AlexNet-CIFAR10 example in `/examples/alexnet` by executing the following:

```
python ./examples/alexnet/runner.py
```

Note the only changes to make to the training script are to import the following:

```
from compressnn.CompressNN import CompressNNModel
from compressnn.utils import contiguous_float32_check
```

and convert model as follows:

```
model = CompressNNModel(model,"cuszp","rel",1e-3,contiguous_float32_check,True)
```

## Notes
CompressNN is constantly evolving and will support more compressors, more customization, and more compression targets in the future. Please email mkshah5@ncsu.edu for suggestions.