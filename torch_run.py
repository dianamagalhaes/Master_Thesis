import argparse
import importlib
import sys
import os
import json
from xml.parsers.expat import model
from sklearn.metrics import log_loss
import torch
from torchvision import transforms, datasets
import pickle
from torch_ava.data import MedNISTDataset, LoaderOperator
from torch_ava.data.get_transformations import DataAugOperator
from torch_ava.torch_utils import TensorboardLoggerOperator, ModelOperator
from torch_ava.engine.trainer import Trainer
from torch_ava.engine.evaluator import Evaluator


def set_import(model_name):

    if "ava_model" in sys.modules.keys():
        submodules = [mod for mod in sys.modules.keys() if mod.startswith("ava_model.")]
        del sys.modules["ava_model"]
        for submodule in submodules:
            del sys.modules[submodule]

    module_path = os.path.join(f"./models/Demo", "__init__.py")
    module_path = os.path.abspath(module_path)

    spec = importlib.util.spec_from_file_location("ava_model", module_path)

    module = importlib.util.module_from_spec(spec)
    sys.modules["ava_model"] = module
    spec.loader.exec_module(module)


def get_configs():

    json_file_path = os.path.join(os.path.dirname(ava_model.__file__), "configs.json")

    with open(json_file_path) as f:
        json_confs = json.load(f)
        return json_confs


def run_model_train(args):

    json_confs = get_configs()

    # Model and Tensorboard Logging
    model_dir = os.path.dirname(ava_model.__file__)

    # Dataset
    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
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

    ch, w, h = train_data[0][0].shape
    inpt_dims = [train_loader.batch_size, ch, w, h]
    print(inpt_dims)
    exit()
    model = torch_model.Network(inpt_dims)
    optim, scheduler = torch_model.get_optimizer(model)
    print(model)

    model_operator = ModelOperator(torch_model.loss, optim, use_cuda=args.gpu)

    logger = TensorboardLoggerOperator(model_dir, labels_index=mednist_data.class_to_idx)
    print(logger.tb_logger)
    print(type(logger.tb_logger))

    trainer_operator = Trainer(model_operator, logger, epochs=json_confs["train"]["epochs"])
    trainer_operator.run_epochs(model, train_loader, val_loader, scheduler)

    
def run_model_eval(args):

    args.model_name
    args.epoch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "model_name", metavar="model_name", type=str, help="Model folder name that will configure the training.",
    )
    parser.add_argument("--gpu", default=False, type=str, help="If available which gpu id it will use (e.g., gpu 0).")
    subparsers = parser.add_subparsers(help="Model Stage.")

    train = subparsers.add_parser("train", help="Training stage")
    train.set_defaults(func=run_model_train)

    test = subparsers.add_parser("test", help="Evaluation stage")
    test.set_defaults(func=run_model_eval)
    test.add_argument("-e", "--epoch", required=True, type=int, help="Epoch number.")

    args = parser.parse_args()
    set_import(args.model_name)

    import ava_model
    from ava_model import torch_model

    args.func(args)
