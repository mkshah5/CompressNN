import sys
sys.path.append("../")

import torch
import torch.nn as nn

from compressnn.utils import contiguous_float32_check

'''
Wraps a PyTorch module and compresses activations

__init__ Arguments:
- compressor: compressor name (str)
- err_mode: cuSZp error bound mode (str)
- err_bound: cuSZp error bound (float)
- compress_check: function header that returns true when an activation should be compressed (function)
- free_space: Frees original and compressed data appropriately (bool)
- get_debug: Print debug information (bool)
'''
class CompressNNModel(nn.Module):
    def __init__(self, model, compressor="cuszp", err_mode="rel", err_bound=1e-3, compress_check=contiguous_float32_check, free_space=True, get_debug=False):
        super(CompressNNModel, self).__init__()
        self.internal_model = model
        if compressor=="cuszp":
            from compressnn.compressors.cuszpcompress import CUSZpCompressor
            self.compressor = CUSZpCompressor(compressor, err_mode, err_bound, compress_check, free_space, get_debug)
        elif compressor=="cpu":
            from compressnn.compressors.cpucompress import CPUCompressor
            self.compressor = CPUCompressor(compressor, compress_check)
        else:
            from compressnn.compressors.cpucompress import CPUCompressor
            self.compressor = CPUCompressor(compressor, compress_check)


    def forward(self, x):
        self.compressor.reset_tcount()
        with torch.autograd.graph.saved_tensors_hooks(
            self.pack_hook, self.unpack_hook
        ):
            return self.internal_model(x)
    
    def pack_hook(self, x):
        return self.compressor.compress(x)
        
    def unpack_hook(self, x):
        return self.compressor.decompress(x)
