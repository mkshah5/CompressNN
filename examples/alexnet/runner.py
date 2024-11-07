#import sys
#sys.path.append("../")

import argparse

import torch
from alexnet import CIFAR10, AlexNetBaseline

import compressnn

from compressnn.CompressNN import CompressNNModel
from compressnn.utils import contiguous_float32_check

EPOCHS = 30

def train(model, train_loader, valid_loader, loss_function, optimizer):
    # Train the model
    model = model.cuda()
    total_step = len(train_loader)
    for epoch in range(EPOCHS):
        avg_loss = 0.0
        
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            
            outputs = model(x)
            
            loss = loss_function(outputs,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.mean()
            del loss
            torch.cuda.empty_cache()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, EPOCHS, i + 1, total_step,
                                                                     avg_loss / (i + 1)))

        test_acc = 0.0
        valid_steps = len(valid_loader)
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for i,(x, y) in enumerate(valid_loader):
           
                x = x.cuda()
                y = y.cuda()

                outputs = model(x)
                loss = loss_function(outputs,y)
                                
                total_loss += loss.mean()
                total += y.size(0)
                correct += (outputs == y).sum()

                del loss
                torch.cuda.empty_cache()
            test_acc = 100.0 * correct / total
            print('Test Accuracy: {:.2f}'.format(test_acc),
                  ' Loss: {:.4f}'.format(total_loss.item() / valid_steps))

def main():
    model = AlexNetBaseline(10)
    train_loader, valid_loader = CIFAR10(50)
   
    model = CompressNNModel(model,"cuszp","rel",1e-3,contiguous_float32_check,True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4)
    model = model.cuda()
    
    train(model, train_loader, valid_loader, loss_fn, optimizer)

if __name__ == '__main__':
    main()
