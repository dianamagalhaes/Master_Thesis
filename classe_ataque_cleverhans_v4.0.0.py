from cgi import test
import json
from easydict import EasyDict
import joblib
import os
import importlib
import sys
import torch.optim as optim

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Union

import torch
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F

# CleverHans Lib v 4.0.0
from cleverhans.utils import AccuracyReport
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


# -- Custom Libs
from torch_ava.data.get_transformations import DataAugOperator
from torch_ava.engine.evaluator import Evaluator
from torch_ava.data import MedNISTDataset
from torch_ava.data.gen_dataset_loader import LoaderOperator
from torch_ava.engine.evaluator import Evaluator
from torch_ava.torch_utils.operators import ModelOperator
from models.Demo.torch_model import Network


def set_module_import(model_name):

    if "ava_model" in sys.modules.keys():
        submodules = [mod for mod in sys.modules.keys() if mod.startswith("ava_model.")]
        del sys.modules["ava_model"]
        for submodule in submodules:
            del sys.modules[submodule]

    module_path = os.path.join(f"./models/{model_name}", "__init__.py")
    module_path = os.path.abspath(module_path)

    spec = importlib.util.spec_from_file_location("ava_model", module_path)

    module = importlib.util.module_from_spec(spec)
    sys.modules["ava_model"] = module
    spec.loader.exec_module(module)


def get_configs(json_file_path: str) -> dict:

    with open(json_file_path) as f:
        json_confs = json.load(f)
        return json_confs


class Ataque:
    def __init__(self, model, path: str):
        self.model = model
        self.path = path

    def load_model(self, json_confs: dict):

        # Loading the Epoch correctly
        # TODO Check if the variable model needs to be used at the call Ataque.load_dataset()
        # Also this function is performing operations beyond the scope of its name. Maybe it could be better to
        # separate them.
        model = torch.load(self.path)
        data_load = Ataque.load_dataset(json_confs)
        print("Model successfully load!")

    def load_epoch(self, model):
        x = LoaderOperator(torch_dset=path)
        train_dl = x.get_loader(mode=train, torch_dset=path, batch_size=50)
        test_dl = x.get_loader(mode=test, torch_dset=path, batch_size=50)

        return train_dl, test_dl

    def load_dataset(json_confs: dict, path_dataset: str = "./MedNIST/"):
        # Dataset
        # Load Data Transformation Pipelines
        train_data_aug = DataAugOperator()
        train_data_aug.set_pipeline(json_confs["train"]["transformations"])
        val_data_aug = DataAugOperator()
        val_data_aug.set_pipeline(json_confs["val"]["transformations"])
        # Load Dataset
        mednist_data = datasets.ImageFolder(root=path_dataset)
        train_data = MedNISTDataset(mednist_data, train_data_aug.get_pipeline())
        val_data = MedNISTDataset(mednist_data, val_data_aug.get_pipeline())
        data_loader = LoaderOperator(train_data)
        train_loader = data_loader.get_loader("train", train_data, json_confs["train"]["batch_size"])
        val_loader = data_loader.get_loader("val", val_data, json_confs["val"]["batch_size"])

        ch, w, h = train_data[0][0].shape
        inpt_dims = [train_loader.batch_size, ch, w, h]
        print("Input Dimensions:", inpt_dims)
        model = torch_model.Network(inpt_dims)
        optimizer, scheduler = torch_model.get_optimizer(model)
        print(model)

        # TODO add the possibility to define which hardware will run the train
        model_operator = ModelOperator(torch_model.loss, optim, use_cuda="0")
        model.to(model_operator.get_device())

        # Train vanilla model
        model.train()
        for epoch in range(1, 10 + 1):
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(model_operator.get_device()), y.to(model_operator.get_device())

                optimizer.zero_grad()
                out = model(x)
                loss = model_operator.compute_loss(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print("epoch: {}/{}, train loss: {:.3f}".format(epoch, 10, train_loss))

        # Evaluate on clean and adversarial data

        # Total eps of the adversarial attack
        # TODO in the future we might want to explore this variable, thus we need to make it configurable
        adva_eps = 0.3

        model.eval()
        report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
        for x, y in val_loader:
            x, y = x.to(model_operator.get_device()), y.to(model_operator.get_device())
            x_fgm = fast_gradient_method(model, x, adva_eps, np.inf)
            x_pgd = projected_gradient_descent(model, x, adva_eps, 0.01, 40, np.inf)
            _, y_pred = model(x).max(1)  # model prediction on clean examples
            _, y_pred_fgm = model(x_fgm).max(1)  # model prediction on FGM adversarial examples
            _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples
            report.nb_test += y.size(0)
            report.correct += y_pred.eq(y).sum().item()
            report.correct_fgm += y_pred_fgm.eq(y).sum().item()
            report.correct_pgd += y_pred_pgd.eq(y).sum().item()
        print("test acc on clean examples (%): {:.3f}".format(report.correct / report.nb_test * 100.0))
        print("test acc on FGM adversarial examples (%): {:.3f}".format(report.correct_fgm / report.nb_test * 100.0))
        print("test acc on PGD adversarial examples (%): {:.3f}".format(report.correct_pgd / report.nb_test * 100.0))


if __name__ == "__main__":

    import platform

    model_name = "AdvaDemo_CleverHans_4.0.0"

    if platform.node() == "nea138-lt":  # Adriano Pinto's machine name
        json_file_path = f"/home/apinto/Documents/projects/msc_thesis_DianaMag/models/{model_name}/configs.json"
    else:
        json_file_path = f"/home/diana-dtx/Desktop/Master_Thesis/models/{model_name}/configs.json"
    path = "models/Demo/LOGS/models/nnet_epoch_9.pt"
    model = {}
    train = None
    train_dl = None
    test_dl = None

    # Let's start the engine environment
    set_module_import(model_name)

    import ava_model
    from ava_model import torch_model

    json_confs = get_configs(json_file_path)

    ataque = Ataque(model, path)
    ataque.load_model(json_confs)
    train_dl, test_dl = ataque.load_epoch(model)

