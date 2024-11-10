import sys
sys.path.append("../")

import torch
import torch.nn as nn

class Tracer():
    def __init__(self, model):
        super(Tracer, self).__init__()
        self.internal_model = model
        self.map = dict()

    def trace(self):
        i = 0
        for name, layer in self.internal_model.named_modules():
            if not isinstance(layer, nn.Sequential):
                if i != 0:
                    self.map[i] = type(layer).__name__
        return self
    def get_tensor_trace(self):
        return self.map