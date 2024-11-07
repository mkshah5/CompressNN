import torch

class CompressedElement():
    def __init__(self, compressed_data, uncompressed_elements, original_shape, datatype, compress_type="cuszp"):
        super(CompressedElement, self).__init__()
        self.compressed_data = compressed_data
        self.original_shape = original_shape
        self.dtype = datatype
        self.compress_type = compress_type
        self.uncompressed_elements = uncompressed_elements

def contiguous_float32_check(x):
    return x.is_contiguous() and x.dtype == torch.float32