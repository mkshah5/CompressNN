import sys
sys.path.append("../")

import torch
import torch.nn as nn
import json

from compressnn.utils import contiguous_float32_check, CompressedElement
from compressnn.compressors.compressor import Compressor

class Composer():
    def __init__(self, compress_config_path, compress_check, trace, free_space, get_debug):
        super(Composer, self).__init__()        
        with open(compress_config_path) as json_file:
            self.compress_config = json.load(json_file)
        self.tcount = 0
        self.compress_check = compress_check
        self.trace = trace
        self.compressor_trace = dict()

        for t in self.trace.keys():
            if "layer_type" in self.compress_config:
                if self.trace[t] in self.compress_config["layer_type"]:
                    self.compressor_trace[t] = get_compressor(self.compress_config["layer_type"][self.trace[t]], compress_check, free_space, get_debug)
            
            if "layer_number" in self.compress_config:
                if str(t) in self.compress_config["layer_number"]:
                    self.compressor_trace[t] = get_compressor(self.compress_config["layer_number"][str(t)], compress_check, free_space, get_debug)

            if t not in self.compressor_trace:
                self.compressor_trace[t] = get_compressor(self.compress_config["default"], compress_check, free_space, get_debug)
    ### Moves input tensor x to CPU if passes compress_check
    def compress_pass(self, x):
        data = self.compressor_trace[self.tcount].compress(x)
        if isinstance(data, CompressedElement):
            self.tcount+=1    
        return (data, self.tcount-1)
    ### Moves input tensor x to GPU if passes compress_check
    def decompress_pass(self, x):
        if isinstance(x, tuple):
            (x, tcount) = x
        if isinstance(x, CompressedElement):
            data = self.compressor_trace[tcount].decompress(x)
        else:
            data = x
        return data
        
def get_compressor(config, compress_check, free_space, get_debug):
    if config["compressor"]=="cuszp":
            from compressnn.compressors.cuszpcompress import CUSZpCompressor
            compressor = CUSZpCompressor(config["compressor"], config["error_mode"], float(config["error_bound"]), compress_check, free_space, get_debug)
    elif config["compressor"]=="cpu":
        from compressnn.compressors.cpucompress import CPUCompressor
        compressor = CPUCompressor(config["compressor"], compress_check)
    else:
        from compressnn.compressors.cpucompress import CPUCompressor
        compressor = CPUCompressor(config["compressor"], compress_check)
    return compressor
