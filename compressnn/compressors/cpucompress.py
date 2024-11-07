import sys
sys.path.append("..")
import torch
import cuszp

from utils import contiguous_float32_check, CompressedElement
from compressor import Compressor

class CPUCompressor(Compressor):
    def __init__(self, name="", compress_check=contiguous_float32_check):
        super(Compressor, self).__init__(name)        
        self.compress_check = compress_check

    def increment_tcount(self):
        self.tensor_count += 1
    
    def reset_tcount(self):
        self.tensor_count = 0

    def compress(self, x):
        data = None
        if self.compress_check(x):
            data = CompressedElement(x.cpu(), x.numel(), x.shape, x.dtype, self.scheme)
        else:
            data = x
            
        self.increment_tcount()
        return data
    
    def decompress(self, x):
        data = None
        if isinstance(x, CompressedElement):
            assert x.compress_type == "cpu", "Decompressing non-CPU compressed data with CPUCompressor!"
            decompressed = x.compressed_data.cuda()
            data = decompressed.reshape(x.original_shape)
        else:
            data = x

        return data