import sys
sys.path.append("../")

import torch
import torch.nn as nn

class TracerModel(nn.Module):
    def __init__(self, model):
        super(TracerModel, self).__init__()
        self.internal_model = model
        self.count = 0
    def forward(self, x):
        with torch.autograd.graph.saved_tensors_hooks(
            self.pack_hook, self.unpack_hook
        ):
            return self.internal_model(x)
    
    def pack_hook(self, x):
        self.count+=1
        return x
        
    def unpack_hook(self, x):
        return x

class Tracer():
    def __init__(self, model, input_shape, output_shape, output_dtype, loss_function):
        super(Tracer, self).__init__()
        self.internal_model = model
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self.loss_function = loss_function
        self.map = dict()
        self.counter = 0

    def trace(self):
        for name, layer in self.internal_model.named_modules():
            print(name,layer)
        tmodel = TracerModel(self.internal_model)
        outputs = tmodel(torch.randn(self.input_shape))
            
        loss = self.loss_function(outputs,torch.randn(self.output_shape).to(self.output_dtype))

        loss.backward()
        print(tmodel.count)
