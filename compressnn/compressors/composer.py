import sys
sys.path.append("../")

import torch
import torch.nn as nn
import json

from compressnn.utils import CompressedElement

'''
Composes the set of compressors used given a model trace from Tracer() and a compression configuration

__init__ Arguments:
- config_path: Path to a compression JSON configuration file (str)
- compress_check: function header that returns true when an activation should be compressed (function)
- free_space: Frees original and compressed data appropriately (bool)
- get_debug: Print debug information (bool)
'''

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
    ### Performs the compression pass
    def compress_pass(self, x):
        data = self.compressor_trace[self.tcount].compress(x)
        if isinstance(data, CompressedElement):
            self.tcount+=1    
        return (data, self.tcount-1)
    ### Performs the decompression pass
    def decompress_pass(self, x):
        if isinstance(x, tuple):
            (x, tcount) = x
        if isinstance(x, CompressedElement):
            data = self.compressor_trace[tcount].decompress(x)
        else:
            data = x
        return data
        
### Returns a compressor given a compression configuration
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
