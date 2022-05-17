from collections import OrderedDict
import json
from pickletools import optimize
import torch
import joblib
import os
import importlib
import sys
import torch.optim as optim
from yaml import load

path= 'models/Demo/LOGS/models/nnet_epoch_9.pt'
model={}

class ataque: 
    def __init__(self, model, path):
        self.model= model
        self.path= path

    def load_model(model, path):    
        
        #Avoiding the ava_module not found:
        module_path = os.path.join(f"./models/Demo", "__init__.py")
        module_path = os.path.abspath(module_path)
        spec = importlib.util.spec_from_file_location("ava_model", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["ava_model"] = module
        spec.loader.exec_module(module)
        
        #Loading the Epoch correctly
        model=torch.load(path)
        model.eval()
        print("Epoch successfully load!")
        #print(model)
        
        #Model State after load
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])


ataque.load_model(model,path)

