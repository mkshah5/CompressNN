class Compressor():
    def __init__(self, name=""):
        super(Compressor, self).__init__()
        self.name = name
        self.tensor_count = 0
    
    def increment_tcount(self):
        self.tensor_count += 1
    
    def reset_tcount(self):
        self.tensor_count = 0

    ### Override in subclasses
    def compress(self, x):
        pass

    ### Override in subclasses
    def decompress(self, x):
        pass