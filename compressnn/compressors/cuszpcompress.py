import sys
sys.path.append("..")
import torch
import cuszp

from utils import contiguous_float32_check, CompressedElement
from compressor import Compressor

class CUSZpCompressor(Compressor):
    def __init__(self, name="", err_mode="rel", err_bound=1e-3, compress_check=contiguous_float32_check, free_space=True):
        super(Compressor, self).__init__(name)        
        self.err_mode = err_mode
        self.err_bound = err_bound
        self.compress_check = compress_check
        self.free_space = free_space

    def increment_tcount(self):
        self.tensor_count += 1
    
    def reset_tcount(self):
        self.tensor_count = 0

    def compress(self, x):
        data = None
        if self.compress_check(x):
            data = cuszp.compress(x, self.err_bound, self.err_mode)
            data = CompressedElement(data, x.numel(), x.shape, x.dtype, self.scheme)
            if self.free_space:
                del x
                torch.cuda.empty_cache()
        else:
            data = x
            
        self.increment_tcount()
        return data
    
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