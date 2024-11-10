import sys
sys.path.append("..")
import torch
import cuszp

from compressnn.utils import contiguous_float32_check, CompressedElement
from compressnn.compressors.compressor import Compressor

'''
PyTorch-based implementation of the cuSZp compressor from https://github.com/szcompressor/cuSZp.
Overrides compress() and decompress() from parent class Compressor

__init__ Arguments:
- name: compressor name (str)
- err_mode: cuSZp error bound mode (str)
- compress_check: function header that returns true when an activation should be compressed (function)
- free_space: Will delete original data during compression and compressed data during decompression if True (bool)
- get_debug: Prints debug information (bool)

'''
class CUSZpCompressor(Compressor):
    def __init__(self, name="", err_mode="rel", err_bound=1e-3, compress_check=contiguous_float32_check, free_space=True, get_debug=False):
        super().__init__(name)        
        self.err_mode = err_mode
        self.err_bound = err_bound
        self.compress_check = compress_check
        self.free_space = free_space
        self.get_debug=get_debug

    ### Compresses input tensor x if passes compress_check
    def compress(self, x):
        data = None
        if self.compress_check(x):
            data = cuszp.compress(x, self.err_bound, self.err_mode)
            data = CompressedElement(data, x.numel(), x.shape, x.dtype, self.name)
            if self.get_debug:
                print("Original Size (B): "+str(x.numel()*x.element_size())+", Compressed Size (B): "+str(data.compressed_data.numel()*data.compressed_data.element_size()))
            if self.free_space:
                del x
                torch.cuda.empty_cache()
            
        else:
            data = x
            
        return data
    
    ### Decompresses input tensor x if CompressedElement that was used by this compressor
    def decompress(self, x):
        data = None
        if isinstance(x, CompressedElement):
            assert x.compress_type == "cuszp", "Decompressing non-cuSZp compressed data with cuSZp!"
            decompressed = cuszp.decompress(
                        x.compressed_data,
                        x.uncompressed_elements,
                        x.compressed_data.numel()*x.compressed_data.element_size(),
                        self.err_bound,
                        self.err_mode,
                    )
            
            data = decompressed.reshape(x.original_shape)
            if self.free_space:
                del x.compressed_data
                del x
                torch.cuda.empty_cache()
        else:
            data = x

        return data
