import sys
sys.path.append("../")

import torch
import torch.nn as nn

from utils import contiguous_float32_check

class CompressNNModel(nn.Module):
    def __init__(self, model, compressor="cuszp", err_mode="rel", err_bound=1e-3, compress_check=contiguous_float32_check, free_space=True):
        super(CompressNNModel, self).__init__()
        self.internal_model = model
        if compressor=="cuszp":
            from compressors.cuszpcompress import CUSZpCompressor
            self.compressor = CUSZpCompressor(compressor, err_mode, err_bound, compress_check, free_space)
        elif compressor=="cpu":
            from compressors.cpucompress import CPUCompressor
            self.compressor = CPUCompressor(compressor, compress_check)
        else:
            from compressors.cpucompress import CPUCompressor
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
