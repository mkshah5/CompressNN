import sys
sys.path.append("..")
import torch

from compressnn.utils import contiguous_float32_check, CompressedElement
from compressnn.compressors.compressor import Compressor

'''
Compressor that does no compression. Placeholder for no compression activations.

__init__ Arguments:
- name: compressor name (str)
- compress_check: function header that returns true when an activation should be compressed (function)

'''

class NoCompressor(Compressor):
    def __init__(self, name=""):
        super().__init__(name)        

    ### Moves input tensor x to CPU if passes compress_check
    def compress(self, x):            
        return x
    
    ### Moves input tensor x to GPU if passes compress_check
    def decompress(self, x):
        return x
