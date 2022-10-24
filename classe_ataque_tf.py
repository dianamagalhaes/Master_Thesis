import argparse
import glob
import importlib
import json
import os
import pandas as pd
import platform
from tqdm import tqdm
import numpy as np
import sys

# DL libraries
import torch
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)
# tf.compat.v1.Session()


# Classification performance metrics
from sklearn.metrics import classification_report

# CleverHans Lib v 3.1.0
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2

# ---
from torch_ava.data import CCAB_Dataset


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
        "Fast_Gradient_Method": {
            "call": fast_gradient_method,
            "kwargs": {"eps": 0.3, "norm": np.inf},
        },
        "Projected_Gradient_Descent": {
            "call": projected_gradient_descent,
            "kwargs": {"eps": 0.3, "eps_iter": 0.01, "nb_iter": 40, "norm": np.inf},
        },
        "Carlini_Wagner_L2": {"call": carlini_wagner_l2, "kwargs": {"n_classes": 6}},
    }

    def __init__(
        self,
        json_confs: dict,
    ):

        self.json_confs = json_confs

        # self.session = tf.compat.v1.Session()
        self.graph = tf.get_default_graph()

    def get_dataset_loader(self, cardiomr_base_dir: str, dataframe_path: str):

        torch_dset = CCAB_Dataset(cardiomr_base_dir=cardiomr_base_dir, dataframe_path=dataframe_path)

        torch_loader = torch.utils.data.DataLoader(
            torch_dset,
            batch_size=self.json_confs["test"]["batch_size"],
            shuffle=False,
        )

        return torch_loader

    def load_tf_model(self, model_weights_path: str):

        # set_session(self.session)
        model = load_model(model_weights_path)
        print("Model successfully load!")

        print(model.summary())

        return model

    def eval_clean_samples(self, model_name: str, model: object, torch_loader: object):
        # Evaluate on clean data
        y_true, y_pred = [], []

        with tqdm(torch_loader, unit="batch") as prog_torch_loader:
            for x, y in prog_torch_loader:
                x = x.cpu().detach().numpy().astype(np.float32)
                y = y.cpu().detach().numpy()

                pred = model.predict(x, verbose=0).argmax(1).tolist()

                y_true += y.tolist()
                y_pred += pred

            report = classification_report(
                y_true, y_pred, digits=4, target_names=self.json_confs["target_labels_name"], output_dict=True
            )
            df = pd.DataFrame(report).transpose()
            df.to_csv(f"models/{model_name}/LOGS/Results/clean_samples.csv")

    def eval_attack(self, model_name: str, model: object, attack_name: str, torch_loader: object):

        y_true, y_pred_attacked = [], []

        if attack_name in Ataque.adversarial_attacks:
            adva_attack = Ataque.adversarial_attacks[attack_name]

        else:
            raise ValueError("Please provide a valid and supported Attack")

        from PIL import Image
        import cv2

        with tqdm(torch_loader, unit="batch") as prog_torch_loader:
            for x_test, y_test in prog_torch_loader:
                x_test = x_test.cpu().detach().numpy().astype(np.float32)
                y_test = y_test.cpu().detach().numpy()

                # image_array = cv2.normalize(x_test[0, :, :, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # im = Image.fromarray(image_array)
                # im.save("input.png")

                adv_x = adva_attack["call"](model, x_test, **adva_attack["kwargs"])
                adv_x_numpy = adv_x.numpy()

                # adv_img = cv2.normalize(adv_x_numpy[0, :, :, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # adv_im = Image.fromarray(adv_img)
                # adv_im.save("result.png")

                # print("Original == Adversarial?", (adv_x_numpy[0, :, :, 0] == x_test[0, :, :, 0]).all())

                pred_attacked = model.predict(adv_x_numpy, verbose=0).argmax(1).tolist()

                y_pred_attacked += pred_attacked
                y_true += y_test.tolist()

        attack_report = classification_report(
            y_true, y_pred_attacked, digits=4, target_names=self.json_confs["target_labels_name"], output_dict=True
        )
        attack_acro = attack_name.replace(" ", "").lower()
        df = pd.DataFrame(attack_report).transpose()
        df.to_csv(f"models/{model_name}/LOGS/Results/{attack_acro}_attacked_test_samples.csv")


if __name__ == "__main__":

    # Package configurations
    pckg_confs = open("pckg_confs.json")
    pckg_confs = json.load(pckg_confs)

    # Check for gpu availability
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cuda_devices = torch.cuda.device_count()
    cuda_codenames = [f"cuda:{idx}" for idx in range(cuda_devices)]

    # Overall configs

    if platform.node() == "nea138-lt":
        cardiomr_base_dir = "/home/apinto/Documents/repos/cardiomr_dl/"
    else:
        cardiomr_base_dir = "/mnt/SSD_Storage/ai4cmr/cardiomr_dl/"

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

    parser.add_argument(
        "--attack_name",
        required=True,
        type=str,
        help="Name of the Adversarial Attack",
        choices=[
            "None",
            "Fast_Gradient_Method",
            "Projected_Gradient_Descent",
            "Sparse_L1_Descent",
            "Carlini_Wagner_L2",
            "Hop_Skip_Jump",
        ],
    )

    args = parser.parse_args()

    # Let's start the engine environment
    set_module_import(args.model_name)

    import ava_model
    from ava_model import torch_model

    model_configs_path = os.path.abspath(os.path.join(pckg_confs["MODELS_DIR"], f"{args.model_name}/configs.json"))
    model_configs = get_model_configs(model_configs_path)
    os.makedirs(f"models/{args.model_name}/LOGS/Results/", exist_ok=True)

    ataque = Ataque(json_confs=model_configs)

    dataframe_path = os.path.abspath(
        os.path.join(pckg_confs["MODELS_DIR"], "SA_Classification_AI4MED/EXTRA_DETAILS/test_dataframe.csv")
    )
    torch_loader = ataque.get_dataset_loader(cardiomr_base_dir, dataframe_path)

    # Pipeline for an already trained model

    model_weights_path = glob.glob(f"models/{args.model_name}/LOGS/models/*")[0]
    model = ataque.load_tf_model(model_weights_path)

    if args.attack_name == "None":

        ataque.eval_clean_samples("SA_Classification_AI4MED", model, torch_loader)
    else:
        ataque.eval_attack("SA_Classification_AI4MED", model, "Fast_Gradient_Method", torch_loader)
