import sys
sys.path.append("../")

import torch
import torch.nn as nn
import copy
'''
Traces the model to retrieve a mapping of operator number to operator name

__init__ Arguments:
- model: torch.nn.Module to trace (torch.nn.Module)
'''
class TracerModel(nn.Module):
    def __init__(self, model):
        super(TracerModel, self).__init__()
        self.internal_model = copy.deepcopy(model)
        self.map = dict()
        self.tcount = 0
        
    def forward(self, x):
        return self.internal_model(x)

    def destroy(self):
        del self.internal_model
        return

class Tracer():
    def __init__(self, model, batch_size, input_shapes):
        super(Tracer, self).__init__()
        self.internal_model = TracerModel(model)
        self.map = dict()
        self.module_order = []
        self.batch_size = batch_size
        self.input_shapes = input_shapes
        for name, module in self.internal_model.named_modules():
            module.register_forward_hook(self.hook)
    def hook(self,module, input, output):
        #if input[0].shape[0] == self.batch_size:
        self.module_order.append(module.__class__.__name__)

    def trace(self):
        inputs = []
        for shape in self.input_shapes:
            inputs.append(torch.randn(shape))
        inputs = tuple(inputs)
        y = self.internal_model(*inputs)
        i = 0
        for i in range(len(self.module_order)):
            self.map[i] = self.module_order[i]
        return self
    def get_tensor_trace(self):
        return self.map
    def destroy(self):
        self.internal_model.destroy()
