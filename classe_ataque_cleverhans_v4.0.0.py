from cgi import test
import json
import statistics
from easydict import EasyDict
import joblib
import os
import importlib
import sys
import glob
import argparse


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

# from cleverhans.torch.attacks.lbfgs import LBFGS
from torch_ava.data.get_transformations import DataAugOperator
from torch_ava.data import MedNISTDataset, LoaderOperator
from torch_ava.torch_utils import TensorboardLoggerOperator, ModelOperator
from torch_ava.engine.trainer import Trainer

torch.cuda.empty_cache()


def set_module_import(model_name: str):
    """This utility function imports the model folder as an additional module of this script.
    Allowing it to gather all the necessary information for train/val/test and also to attack the model.

    Args:
        model_name (str): String that matches the codename of the model ought to be imported as a module.
    """
    if "ava_model" in sys.modules.keys():
        submodules = [mod for mod in sys.modules.keys() if mod.startswith("ava_model.")]
        del sys.modules["ava_model"]
        for submodule in submodules:
            del sys.modules[submodule]

    module_path = os.path.abspath(os.path.join(f"./models/{model_name}", "__init__.py"))

    spec = importlib.util.spec_from_file_location("ava_model", module_path)

    module = importlib.util.module_from_spec(spec)
    sys.modules["ava_model"] = module
    spec.loader.exec_module(module)


def get_model_configs(json_file_path: str) -> dict:

    with open(json_file_path) as f:
        return json.load(f)


class Ataque:

    adversarial_attacks = {
        "Fast Gradient Method": {"call": fast_gradient_method, "kwargs": {"eps": 0.3, "norm": np.inf}},
        "Projected Gradient Descent": {
            "call": projected_gradient_descent,
            "kwargs": {"eps": 0.3, "eps_iter": 0.01, "nb_iter": 40, "norm": np.inf},
        },
        "Sparse L1 Descent": {"call": sparse_l1_descent, "kwargs": {}},
        "Carlini Wagner L2": {"call": carlini_wagner_l2, "kwargs": {"n_classes": 6}},
        "Hop Skip Jump": {"call": hop_skip_jump_attack, "kwargs": {"norm": np.inf}},
    }

    def __init__(self, json_confs: dict, dataset_path: str = "./MedNIST/"):

        self.json_confs = json_confs
        self.dataset_path = dataset_path

    @staticmethod
    def _get_loader(mode: str, main_data_loader: object, main_torch_dset: object, json_confs: dict):

        data_trfm = DataAugOperator()
        data_trfm.set_pipeline(json_confs[mode]["transformations"])
        torch_dset = MedNISTDataset(main_torch_dset, data_trfm.get_pipeline())
        torch_loader = main_data_loader.get_loader(mode, torch_dset, json_confs[mode]["batch_size"])

        return torch_dset, torch_loader

    def load_dataset(self):

        # Load Dataset
        mednist_data = datasets.ImageFolder(root=self.dataset_path)
        self.labels_to_idx = mednist_data.class_to_idx
        data_loader = LoaderOperator(mednist_data)

        torch_dset, torch_train_loader = self._get_loader("train", data_loader, mednist_data, self.json_confs)
        _, torch_val_loader = self._get_loader("val", data_loader, mednist_data, self.json_confs)
        _, torch_test_loader = self._get_loader("test", data_loader, mednist_data, self.json_confs)

        train_features, _ = next(iter(torch_train_loader))
        print(f"Feature batch shape: {train_features.size()}")
        return torch_dset, torch_train_loader, torch_val_loader, torch_test_loader

    def load_torch_model(self, model_weights_path: str):

        model = torch.load(model_weights_path)

        if model_weights_path.endswith(".pth"):
            print("The model saving only considered the dictionary of weights. Not the whole model.")

            torch_model.model.load_state_dict(model)
            model = torch_model.model

        print("Model successfully load!")

        return model

    @staticmethod
    def set_model_to_train(torch_train_dset, torch_train_loader):
        ch, w, h = torch_train_dset[0][0].shape
        inpt_dims = [torch_train_loader.batch_size, ch, w, h]
        print("Input Dimensions:", inpt_dims)
        if hasattr(torch_model, "Network"):
            model = torch_model.Network(inpt_dims)
        else:
            model = torch_model.model

        optimizer, scheduler = torch_model.get_optimizer(model)
        print(model)

        return model, optimizer, scheduler

    def train_model(self, model, device, torch_train_loader, torch_val_loader, optimizer, scheduler):

        model_operator = ModelOperator(device=device)
        model_operator.set_loss(torch_model.loss)
        model_operator.set_optimizer(optimizer)
        model.to(model_operator.get_device())

        model_dir = os.path.dirname(ava_model.__file__)
        logger = TensorboardLoggerOperator(model_dir, labels_index=self.labels_to_idx)

        trainer_operator = Trainer(model_operator, logger, epochs=self.json_confs["train"]["epochs"])
        trainer_operator.run_epochs(model, torch_train_loader, torch_val_loader, scheduler)

    def eval_attack(self, model: object, attack_name: str, torch_loader: object, device: str):
        # Evaluate on clean and adversarial data

        model_operator = ModelOperator(device=device)
        device = model_operator.get_device()
        model.to(device)
        model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            for x_val, y_val in torch_loader:
                data, labels = (
                    x_val.to(device),
                    y_val.to(device),
                )

                pred = model(data).argmax(dim=1)
                for i in range(len(pred)):
                    y_true.append(labels[i].item())
                    y_pred.append(pred[i].item())

        if attack_name in Ataque.adversarial_attacks:
            adva_attack = Ataque.adversarial_attacks[attack_name]

        else:
            raise ValueError("Please provide a valid and supported Attack")
        y_pred_attacked = []

        for x, y in torch_loader:
            x, y = x.to(device), y.to(device)
            x_attacked = adva_attack["call"](
                model, x, **adva_attack["kwargs"]
            )  # fast_gradient_method(model, x, adva_eps, np.inf)
            pred_attacked = model(x_attacked).cpu().data.numpy().argmax(axis=1).tolist()
            y_pred_attacked += pred_attacked

        print(
            "\n Classification Performance on clean samples: \n",
            classification_report(y_true, y_pred, digits=4, target_names=self.json_confs["target_labels_name"]),
        )

        print(
            f"\n Classification Performance on {attack_name} Adversarial Attack: \n",
            classification_report(
                y_true, y_pred_attacked, digits=4, target_names=self.json_confs["target_labels_name"]
            ),
        )


def train_exe(ataque, device):

    torch_dset, torch_train_loader, torch_val_loader, _ = ataque.load_dataset()

    model, optim, _ = ataque.set_model_to_train(torch_dset, torch_train_loader=torch_train_loader)
    ataque.train_model(model, device, torch_train_loader, torch_val_loader, optim, None)


def eval_exe(ataque, device):

    _, _, torch_val_loader, torch_test_loader = ataque.load_dataset()
    # Pipeline for an already trained model

    # We will assume that there is only one best model
    # Note that the torch models can either be with .pt or .pth extension.
    # For .pth models, only a dictionary of the weights are store not the whole model!
    model_weights_path = glob.glob(f"models/{args.model_name}/LOGS/models/best*")[0]

    model = ataque.load_torch_model(model_weights_path)
    print("\n Validation set Attacked")
    ataque.eval_attack(model, "Fast Gradient Method", torch_val_loader, device=device)

    print("\n Test set Attacked")
    ataque.eval_attack(model, "Fast Gradient Method", torch_test_loader, device=device)


if __name__ == "__main__":

    # Package configurations
    pckg_confs = open("pckg_confs.json")
    pckg_confs = json.load(pckg_confs)

    # Check for gpu availability
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cuda_devices = torch.cuda.device_count()
    cuda_codenames = [f"cuda:{idx}" for idx in range(cuda_devices)]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        type=str,
        help="Name of the model project folder that will be used in the study.",
    )
    parser.add_argument(
        "--device",
        required=True,
        type=str,
        default="cpu",
        choices=["cpu"] + cuda_codenames,
        help="Device that will execute the script. Defaults to CPU if not specified, "
        "if the user provides a gpu id and there is None available.",
    )
    subparsers = parser.add_subparsers()

    train_routine = subparsers.add_parser("train", help="Train DL model")
    train_routine.set_defaults(func=train_exe)

    eval_routine = subparsers.add_parser("eval", help="Evaluation and Adversarial Attack")
    eval_routine.set_defaults(func=eval_exe)

    args = parser.parse_args()

    # Let's start the engine environment
    set_module_import(args.model_name)

    import ava_model
    from ava_model import torch_model

    model_configs_path = os.path.abspath(os.path.join(pckg_confs["MODELS_DIR"], f"{args.model_name}/configs.json"))
    model_configs = get_model_configs(model_configs_path)

    ataque = Ataque(json_confs=model_configs)

    args.func(ataque, args.device)

