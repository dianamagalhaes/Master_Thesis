from cgi import test
from collections import OrderedDict
from email.errors import NonASCIILocalPartDefect
import json
from pickletools import optimize
import torch
import joblib
import os
import importlib
import sys
import torch.optim as optim
from yaml import load
from torch_ava.data import gen_dataset_loader

import numpy as np

path= 'models/Demo/LOGS/models/nnet_epoch_9.pt'
model={}
train=None
train_dl=None
test_dl=None
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
        
       
    def load_data_from_clean_model(model, BATCH_SIZE=128):
        x = gen_dataset_loader.LoaderOperator(torch_dset=path)
        train_dl= x.get_loader(mode=train, torch_dset=path, batch_size=50)
        test_dl= x.get_loader(mode=test, torch_dset=path, batch_size=50)
        
    


ataque.load_model(model,path)
ataque.load_data_from_clean_model(model)
