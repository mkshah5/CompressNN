import sys
sys.path.append("../")

import torch
import torch.nn as nn
import json

from compressnn.utils import contiguous_float32_check
from compressnn.Tracer import Tracer
from compressnn.compressors.composer import Composer
'''
Wraps a PyTorch module and compresses activations

__init__ Arguments:
- model: torch.nn.Module to integrate compression with (torch.nn.Module)
- compressor: compressor name (str)
- batch_size: Batch size for training (int)
- input_shape: Shape of input (tuple)
- config_path: Path to a compression JSON configuration file (str)
- compress_check: function header that returns true when an activation should be compressed (function)
- free_space: Frees original and compressed data appropriately (bool)
- get_debug: Print debug information (bool)
'''
class CompressNNModel(nn.Module):
    def __init__(self, model, batch_size, input_shape, config_path="./config.json",compress_check=contiguous_float32_check, free_space=True, get_debug=False):
        super(CompressNNModel, self).__init__()
        self.internal_model = model
        self.batch_size = batch_size
    
        tracer = Tracer(self.internal_model, batch_size, input_shape)
        self.trace = tracer.trace().get_tensor_trace()
        tracer.destroy()
        self.composer = Composer(config_path, compress_check, self.trace, free_space, get_debug)
    def forward(self, x):
        self.composer.tcount = 0
        with torch.autograd.graph.saved_tensors_hooks(
            self.pack_hook, self.unpack_hook
        ):
            return self.internal_model(x)
    
    def pack_hook(self, x):
        if x.shape[0] == self.batch_size and x.requires_grad and not isinstance(x, nn.Parameter):
            return self.composer.compress_pass(x)
        else:
            return x
        
    def unpack_hook(self, x):
        return self.composer.decompress_pass(x)
