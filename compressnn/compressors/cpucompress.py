import sys
sys.path.append("..")
import torch

from compressnn.utils import contiguous_float32_check, CompressedElement
from compressnn.compressors.compressor import Compressor

'''
CPU "compressor" that moves data to and from device memory. Does not actually compress the data.
Overrides compress() and decompress() from parent class Compressor

__init__ Arguments:
- name: compressor name (str)
- compress_check: function header that returns true when an activation should be compressed (function)

'''

class CPUCompressor(Compressor):
    def __init__(self, name="", compress_check=contiguous_float32_check):
        super().__init__(name)        
        self.compress_check = compress_check

    ### Moves input tensor x to CPU if passes compress_check
    def compress(self, x):
        data = None
        if self.compress_check(x):
            data = CompressedElement(x.cpu(), x.numel(), x.shape, x.dtype, self.name)
        else:
            data = x
            
        return data
    
    ### Moves input tensor x to GPU if passes compress_check
    def decompress(self, x):
        data = None
        if isinstance(x, CompressedElement):
            assert x.compress_type == "cpu", "Decompressing non-CPU compressed data with CPUCompressor!"
            decompressed = x.compressed_data.cuda()
            data = decompressed.reshape(x.original_shape)
        else:
            data = x

        return data
