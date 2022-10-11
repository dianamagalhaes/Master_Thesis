import argparse
import glob
import importlib
import json
import os
import pandas as pd
import numpy as np
import sys

# DL libraries
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from tensorflow.keras.models import load_model

# Classification performance metrics
from sklearn.metrics import classification_report

# CleverHans Lib v 4.0.0
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack

# ---
from torch_ava.data.get_transformations import DataAugOperator
from torch_ava.data import CCAB_Dataset, LoaderOperator
from torch_ava.torch_utils import TensorboardLoggerOperator, ModelOperator
from torch_ava.engine.trainer import Trainer


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
        "Fast_Gradient_Method": {"call": fast_gradient_method, "kwargs": {"eps": 0.3, "norm": np.inf}},
        "Projected_Gradient_Descent": {
            "call": projected_gradient_descent,
            "kwargs": {"eps": 0.3, "eps_iter": 0.01, "nb_iter": 40, "norm": np.inf},
        },
        "Sparse_L1_Descent": {"call": sparse_l1_descent, "kwargs": {}},
        "Carlini_Wagner_L2": {"call": carlini_wagner_l2, "kwargs": {"n_classes": 6}},
        "Hop_Skip_Jump": {"call": hop_skip_jump_attack, "kwargs": {"norm": np.inf}},
    }

    def __init__(
        self, json_confs: dict,
    ):

        self.json_confs = json_confs

    @staticmethod
    def get_dataset_loader(cardiomr_base_dir: str, dataframe_path: str):

        torch_dset = CCAB_Dataset(cardiomr_base_dir=cardiomr_base_dir, dataframe_path=dataframe_path)
        print("Numpy image array", torch_dset.__getitem__(0)[0], "\n Class Index", torch_dset.__getitem__(0)[1])

        total_samples = len(torch_dset)
        idx = list(range(total_samples))
        sampler = SubsetRandomSampler(idx)

        torch_loader = torch.utils.data.DataLoader(
            torch_dset, batch_size=1, sampler=sampler, num_workers=2, pin_memory=False,
        )

        return torch_loader

    def load_tf_model(self, model_weights_path: str):

        model = load_model(model_weights_path)
        print("Model successfully load!")

        print(model.summary())

        return model

    def eval_attack(self, model_name: str, model: object, attack_name: str, torch_loader: object, device: str):
        # Evaluate on clean and adversarial data

        y_true = []
        y_pred, y_pred_attacked = [], []
        os.makedirs(f"models/{model_name}/LOGS/Results/", exist_ok=True)

        if not os.path.isfile(f"models/{model_name}/LOGS/Results/clean_samples.csv"):

            for x, y in torch_loader:

                pred = model.predict(x).argmax(1)
                for i in range(len(pred)):
                    y_true.append(y[i].item())
                    y_pred.append(pred[i])

            report = classification_report(
                y_true, y_pred, digits=4, target_names=self.json_confs["target_labels_name"], output_dict=True
            )
            df = pd.DataFrame(report).transpose()
            df.to_csv(f"models/{model_name}/LOGS/Results/clean_samples.csv")

        if attack_name in Ataque.adversarial_atacks:
            adva_attack = Ataque.adversarial_atacks[attack_name]

        else:
            raise ValueError("Please provide a valid and supported Attack")

        for x, y in torch_loader:
            x, y = x.to(device), y.to(device)
            x_attacked = adva_attack["call"](model, x, **adva_attack["kwargs"])

            pred_attacked = model.predict(x_attacked).argmax(1).tolist()
            y_pred_attacked += pred_attacked

        attack_report = classification_report(
            y_true, y_pred_attacked, digits=4, target_names=self.json_confs["target_labels_name"], output_dict=True
        )
        attack_acro = attack_name.replace(" ", "").lower()
        df = pd.DataFrame(attack_report).transpose()
        df.to_csv(f"models/{model_name}/LOGS/Results/{attack_acro}_attacked_{set}_samples.csv")


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
    args = parser.parse_args()

    # Let's start the engine environment
    set_module_import(args.model_name)

    import ava_model
    from ava_model import torch_model

    model_configs_path = os.path.abspath(os.path.join(pckg_confs["MODELS_DIR"], f"{args.model_name}/configs.json"))
    model_configs = get_model_configs(model_configs_path)

    ataque = Ataque(json_confs=model_configs)

    cardiomr_base_dir = "/mnt/SSD_Storage/ai4cmr/cardiomr_dl/"
    dataframe_path = os.path.abspath(
        os.path.join(pckg_confs["MODELS_DIR"], "SA_Classification_AI4MED/EXTRA_DETAILS/test_dataframe.csv")
    )

    torch_loader = ataque.get_dataset_loader(cardiomr_base_dir, dataframe_path)

    # Pipeline for an already trained model

    model_weights_path = glob.glob(f"models/{args.model_name}/LOGS/models/*")[0]
    model = ataque.load_tf_model(model_weights_path)

    ataque.eval_attack("SA_Classification_AI4MED", model, "Fast_Gradient_Method", torch_loader, device=args.device)

