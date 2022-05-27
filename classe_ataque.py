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
import time
from torch_ava.data.get_transformations import DataAugOperator
from models.Demo import torch_model
from torch_ava.engine import evaluator
from torchvision import datasets
from torch_ava.data import MedNISTDataset
from torch_ava.data.gen_dataset_loader import LoaderOperator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_configs():

    json_file_path = "/home/diana-dtx/Desktop/Master_Thesis/models/Demo/configs.json"

    with open(json_file_path) as f:
        json_confs = json.load(f)
        return json_confs

def load_data_from_dataset(path_dataset="./MedNIST/"):
    json_confs=get_configs()
    train_data_aug = DataAugOperator()
    

    train_data_aug.set_pipeline(json_confs["train"]["transformations"])

    val_data_aug = DataAugOperator()
    val_data_aug.set_pipeline(json_confs["val"]["transformations"])

    mednist_data = datasets.ImageFolder(root="./MedNIST/")
    train_data = MedNISTDataset(mednist_data, train_data_aug.get_pipeline())
    val_data = MedNISTDataset(mednist_data, val_data_aug.get_pipeline())

    data_loader = LoaderOperator(train_data)
    train_loader = data_loader.get_loader("train", train_data, json_confs["train"]["batch_size"])
    val_loader = data_loader.get_loader("val", val_data, json_confs["val"]["batch_size"])
    return val_loader





class Ataque: 
    def __init__(self, model, path):
        self.model= model
        self.path= path

    def load_model(self, model, path):    
        #Avoiding the ava_module not found:
        module_path = os.path.join(f"./models/Demo", "__init__.py"
        )
        module_path = os.path.abspath(module_path)
        spec = importlib.util.spec_from_file_location("ava_model", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["ava_model"] = module
        spec.loader.exec_module(module)
        
        #Loading the Epoch correctly
        model=torch.load(path)
        #print(model.state_dict())
        data_load= load_data_from_dataset()
        print(data_load)
        print("Epoch successfully load!")
        print(model)
        model= model.eval()
        print(132)
        print(len(next(iter(data_load))[0][0][0][0]))
        print(234)
        trainfeature, trainlabel = next(iter(data_load))
        print(trainfeature, trainlabel)
        imgplot = plt.imshow(trainfeature[0].squeeze())
        output=model(trainfeature)
        print(output.data)
        exit()
        #exit()
        
       
    def load_data_from_clean_model(self, model):
        x = gen_dataset_loader.LoaderOperator(torch_dset=path)
        train_dl= x.get_loader(mode=train, torch_dset=path, batch_size=50)
        test_dl= x.get_loader(mode=test, torch_dset=path, batch_size=50)
       
        return train_dl, test_dl
        
    

if __name__ == '__main__':
    path= 'models/Demo/LOGS/models/nnet_epoch_9.pt'
    model={}
    train=None
    train_dl=None
    test_dl=None
    
    
    ataque = Ataque(model, path)
    ataque.load_model(model, path)
    train_dl, test_dl = ataque.load_data_from_clean_model(model)
    print(train_dl)

    exit()
    val_loader = data_loader.get_loader("val", val_data, json_confs["val"]["batch_size"])

