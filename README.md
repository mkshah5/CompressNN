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
CompressNN wraps your PyTorch networks (`torch.nn.Module` objects) in a `CompressNNModel` (also a `torch.nn.Module`). `CompressNNModel` allows you to specify a compression configuration to compress activation data as well as parameters for that compressor. Currently, CompressNN supports the following compressors:
- **cuSZp**: A lossy floating point compressor for GPU (https://github.com/szcompressor/cuSZp)
- **CPUCompressor**: Transfers activations to and from host memory to save device memory

Check out `compressnn/compressors/cuszpcompress.py` and `compressnn/compressors/cpucompress.py` for source code and arguments required. Given variable `model` is a `torch.nn.Module`, do the following:

```
model = CompressNNModel(model, <batch_size>, <config_path>,<compress_check>, <free_space>, <get_debug>)
```

`model` should now be able to run like any other model, with the integration of compression.

### Compression Configuration
`CompressNNModel` expects a JSON configuration file to specify how activations are compressed. There are three high-level fields:
1. **default**: The default compression scheme to use
2. **layer_type (Optional)**: Specify a compression scheme for a particular layer type, use the torch.nn.Module class name (ie. "Conv2d" for torch.nn.Conv2d)
3. **layer_number (Optional)**: Specify which activation in the network to compress from an index (layers numbered starting with zero index)

Three example configurations files are in the `configs` directory. Ensure that all compressor parameters are supplied to the file.

## Example
Run the AlexNet-CIFAR10 example in `/examples/alexnet` by executing the following:

```
cd ./examples/alexnet/
python runner.py
```

Note the only changes to make to the training script are to import the following:

```
from compressnn.CompressNN import CompressNNModel
from compressnn.utils import contiguous_float32_check
```

and convert model as follows:

```
model = CompressNNModel(model, batch_size, "../../configs/numbercompress.json",contiguous_float32_check,True,False)
```

## Notes
CompressNN is constantly evolving and will support more compressors, more customization, and more compression targets in the future. Please email mkshah5@ncsu.edu for suggestions.