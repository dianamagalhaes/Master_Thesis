from cgi import test
import json
import statistics
from easydict import EasyDict
import joblib
import os
import importlib
import sys


import numpy as np
import time


import torch
from torchvision import datasets


from sklearn.metrics import classification_report

# CleverHans Lib v 4.0.0
from cleverhans.utils import AccuracyReport
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from torch_ava.data.get_transformations import DataAugOperator
from torch_ava.engine.evaluator import Evaluator
from torch_ava.data import MedNISTDataset
from torch_ava.data.gen_dataset_loader import LoaderOperator
from torch_ava.engine.evaluator import Evaluator
from torch_ava.torch_utils.operators import ModelOperator


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
    def __init__(self, model_weights_path: str, dataset_path: str = "./MedNIST/"):

        self.model_weights_path = model_weights_path
        self.dataset_path = dataset_path

    def load_dataset(self, json_confs: dict):
        # --- Dataset

        # Load Data Transformation Pipelines
        train_data_aug = DataAugOperator()
        val_data_aug = DataAugOperator()

        train_data_aug.set_pipeline(json_confs["train"]["transformations"])
        val_data_aug.set_pipeline(json_confs["val"]["transformations"])

        # Load Dataset
        mednist_data = datasets.ImageFolder(root=self.dataset_path)
        train_data = MedNISTDataset(mednist_data, train_data_aug.get_pipeline())
        val_data = MedNISTDataset(mednist_data, val_data_aug.get_pipeline())
        data_loader = LoaderOperator(train_data)
        torch_train_loader = data_loader.get_loader("train", train_data, json_confs["train"]["batch_size"])
        torch_val_loader = data_loader.get_loader("val", val_data, json_confs["val"]["batch_size"])

        # Display image and label.
        train_features, _ = next(iter(torch_train_loader))
        print(f"Feature batch shape: {train_features.size()}")
        return torch_train_loader, torch_val_loader

    def load_torch_model(self):

        model = torch.load(self.model_weights_path)

        if self.model_weights_path.endswith(".pth"):
            print("The model saving only considered the dictionary of weights. Not the whole model.")

            torch_model.model.load_state_dict(model)
            model = torch_model.model

        print("Model successfully load!")

        return model

    @staticmethod
    def set_model(train_data, torch_train_loader):
        ch, w, h = train_data[0][0].shape
        inpt_dims = [torch_train_loader.batch_size, ch, w, h]
        print("Input Dimensions:", inpt_dims)
        model = torch_model.Network(inpt_dims)
        optimizer, scheduler = torch_model.get_optimizer(model)
        print(model)

        return model, optimizer, scheduler

    @staticmethod
    def train_model(model, torch_train_loader, optimizer):
        # TODO add the possibility to define which hardware will run the train
        # Due to lack of hardware use_cuda must be False

        model_operator = ModelOperator(use_cuda=False)
        model_operator.set_loss(torch_model.loss)
        model_operator.set_optimizer(optimizer)
        model.to(model_operator.get_device())

        # Train vanilla model
        model.train()
        for epoch in range(1, 1 + 1):
            train_loss = 0.0
            for x, y in torch_train_loader:
                x, y = x.to(model_operator.get_device()), y.to(model_operator.get_device())

                optimizer.zero_grad()
                out = model(x)
                loss = model_operator.compute_loss(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print("epoch: {}/{}, train loss: {:.3f}".format(epoch, 1, train_loss))

    @staticmethod
    def eval(model, torch_val_loader):
        # Evaluate on clean and adversarial data

        model_operator = ModelOperator(use_cuda=False)
        device = model_operator.get_device()
        model.to(device)

        # Total eps of the adversarial attack
        # TODO in the future we might want to explore this variable, thus we need to make it configurable
        adva_eps = 0.3

        model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            for x_val, y_val in torch_val_loader:
                data, labels = (
                    x_val.to(device),
                    y_val.to(device),
                )
                pred = model(data).argmax(dim=1)
                for i in range(len(pred)):
                    y_true.append(labels[i].item())
                    y_pred.append(pred[i].item())

        print(
            classification_report(
                y_true, y_pred, target_names=["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"], digits=4
            )
        )
        report = EasyDict(
            nb_test=0,
            correct=0,
            correct_fgm=0,
            correct_pgd=0,
            correct_carliniwagnerl2=0,
            correct_sparse=0,
            correct_lbfgs=0,
            correct_hop=0,
        )
        for x, y in torch_val_loader:
            x, y = x.to(device), y.to(device)
            #x_fgm = fast_gradient_method(model, x, adva_eps, np.inf)
            #x_pgd = projected_gradient_descent(model, x, adva_eps, 0.01, 40, np.inf)
            #x_sparse = sparse_l1_descent(model, x)
            #x_carliniwagnerL2 = carlini_wagner_l2(model, x, n_classes=10)
            #x_lbfgs = LBFGS(model)
            x_hop = hop_skip_jump_attack(model, x, np.inf)
            _, y_pred = model(x).max(1)  # model prediction on clean examples
            #_, y_pred_fgm = model(x_fgm).max(1)  # model prediction on FGM adversarial examples
            #_, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples
            #_, y_pred_carliniwagnerl2 = model(x_carliniwagnerL2).max(1)
            #_, y_pred_sparse = model(x_sparse).max(1) # model prediction on Sparse L1 Descent adversarial examples
            #_, y_pred_lbfgs = model(x_lbfgs).max(1) # model prediction on LBFGS adversarial examples
            _, y_pred_hop = model(x_hop).max(1)  # model prediction on Hop Skip Jump adversarial examples
            report.nb_test += y.size(0)
            report.correct += y_pred.eq(y).sum().item()
            #report.correct_fgm += y_pred_fgm.eq(y).sum().item()
            #report.correct_pgd += y_pred_pgd.eq(y).sum().item()
            #report.correct_carliniwagnerl2 += y_pred_carliniwagnerl2.eq(y).sum().item()
            #report.correct_lbfgs += y_pred_lbfgs.eq(y).sum().item()
            #report.correct_sparse += y_pred_sparse.eq(y).sum().item()
            report.correct_hop += y_pred_hop.eq(y).sum().item()
        print("test acc on clean examples (%): {:.3f}".format(report.correct / report.nb_test * 100.0))
        #print("test acc on FGM adversarial examples (%): {:.3f}".format(report.correct_fgm / report.nb_test * 100.0))
        #print("test acc on PGD adversarial examples (%): {:.3f}".format(report.correct_pgd / report.nb_test * 100.0))
        #print("test acc on Carlini Wagner L2 adversarial examples (%): {:.3f}".format(report.correct_carliniwagnerl2 / report.nb_test * 100.0))
        #print("test acc on Sparse L1 Descent adversarial examples (%): {:.3f}".format(report.correct_sparse / report.nb_test * 100.0))
        #print("test acc on LBFGS adversarial examples (%): {:.3f}".format(report.correct_lbfgs / report.nb_test * 100.0))
        print("test acc on Hop Skip Jump adversarial examples (%): {:.3f}".format(report.correct_hop / report.nb_test * 100.0))


if __name__ == "__main__":

    import platform

    model_name = "MedNIST_Dense121_MONAI"
    # Since this model is in pth, only a dictionary of the weights are store not the whole model!
    model_weights_path = f"models/{model_name}/LOGS/models/best_metric_model.pth"

    if platform.node() == "nea138-lt":  # Adriano Pinto's machine name
        json_file_path = f"/home/apinto/Documents/projects/msc_thesis_DianaMag/models/{model_name}/configs.json"
    else:
        json_file_path = f"/home/diana/Desktop/Master_Thesis/models/{model_name}/configs.json"
    model = {}
    train = None
    train_dl = None
    test_dl = None

    # Let's start the engine environment
    set_module_import(model_name)

    import ava_model
    from ava_model import torch_model

    json_confs = get_configs(json_file_path)

    ataque = Ataque(model_weights_path=model_weights_path)

    # Pipeline for an already trained model

    model = ataque.load_torch_model()
    _, torch_val_loader = ataque.load_dataset(json_confs=json_confs)
    ataque.eval(model, torch_val_loader)

