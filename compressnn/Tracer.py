import sys
sys.path.append("../")

import torch
import torch.nn as nn

'''
Traces the model to retrieve a mapping of operator number to operator name

__init__ Arguments:
- model: torch.nn.Module to trace (torch.nn.Module)
'''

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
                    self.map[i-1] = type(layer).__name__
                i+=1
        return self
    def get_tensor_trace(self):
        return self.map
