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
from torch_ava.engine.evaluator import Evaluator
from torch_ava.torch_utils.operators import ModelOperator
from models.Demo.torch_model import Network
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Union
from cleverhans.utils import AccuracyReport
from torch.autograd import Variable
import torch.nn.functional as F
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from cleverhans.model import CallableModelWrapper
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod



def get_configs():

    json_file_path = "/home/diana-dtx/Desktop/Master_Thesis/models/Demo/configs.json"

    with open(json_file_path) as f:
        json_confs = json.load(f)
        return json_confs

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
        data_load= Ataque.load_dataset()
        print("Model successfully load!")


    def load_epoch(self, model):
        x = gen_dataset_loader.LoaderOperator(torch_dset=path)
        train_dl= x.get_loader(mode=train, torch_dset=path, batch_size=50)
        test_dl= x.get_loader(mode=test, torch_dset=path, batch_size=50)
 
        return train_dl, test_dl
        
    def load_dataset(path_dataset="./MedNIST/"):
        # Dataset
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
        report = AccuracyReport()
        total = 0
        correct = 0
        step = 0
        nb_epochs=1
        train_loss = []
        ch, w, h = train_data[0][0].shape
        inpt_dims = [train_loader.batch_size, ch, w, h]
        model = torch_model.Network(inpt_dims)
        optimizer= torch_model.get_optimizer(model)

        train_acc=[]

        fig, ax = plt.subplots()
        for _epoch in range(nb_epochs):
            for xs, ys in train_loader:
                xs, ys = Variable(xs), Variable(ys)
                if torch.cuda.is_available():
                    xs, ys = xs.cuda(), ys.cuda()
                preds = model(xs)
                loss = F.nll_loss(preds, ys)
                loss.backward()  # calc gradients 
                preds_np = preds.cpu().detach().numpy()
                correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
                total += train_loader.batch_size
                step += 1

                if total % 1000 == 0:
                    acc = float(correct) / total
                    print("[%s] Training accuracy: %.2f%%" % (step, acc * 100))
                    train_acc.append(acc)
                    
                    total = 0
                    correct = 0
        acc_graph=ax.plot(range(len(train_acc)), train_acc, label = 'Accuracy %', color = '#fcba03')
        plt.savefig("/home/diana-dtx/Desktop/Master_Thesis/train_accuracy_plot.PNG")

        # Evaluate on clean data
        total = 0
        correct = 0
        for xs, ys in val_loader:
            xs, ys = Variable(xs), Variable(ys)
            if torch.cuda.is_available():
                xs, ys = xs.cuda(), ys.cuda()

            preds = model(xs)
            preds_np = preds.cpu().detach().numpy()

            correct += (np.argmax(preds_np, axis=1) == ys.cpu().detach().numpy()).sum()
            total += len(xs)

        acc = float(correct) / total
        report.clean_train_clean_eval = acc
        print("[%s] Clean accuracy: %.2f%%" % (step, acc * 100))

        # We use tf for evaluation on adversarial data
        sess = tf.compat.v1.Session()
        tf.compat.v1.disable_eager_execution()
        x_op = tf.compat.v1.placeholder(
            tf.float32,
            shape=(
                16,
                3,
                64,
                64,
            ),
        )

        # Convert pytorch model to a tf_model and wrap it in cleverhans
        tf_model_fn = convert_pytorch_model_to_tf(model)
        cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer="logits")

        # Create an FGSM attack
        fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
        fgsm_params = {"eps": 0.3, "clip_min": 0.0, "clip_max": 1.0}
        adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
        adv_preds_op = tf_model_fn(adv_x_op)

        # Run an evaluation of our model against fgsm
        total = 0
        correct = 0
        for xs, ys in val_loader:
            adv_preds_op = tf.compat.v1.placeholder(tf.float32, [None, 2, 2, 1], name='x-input')
            reshaped_x_op = tf.reshape(adv_preds_op, [-1, 4])
            with tf.compat.v1.Session() as sess:
                x = [[[[16], [3]], [[64], [64]]]]
                #print(sess.run(reshaped_x_op, feed_dict={adv_preds_op: x}))
                adv_preds = sess.run(reshaped_x_op, feed_dict={adv_preds_op: x})
                correct += (np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy()).sum()
                total += val_loader.batch_size

        acc = float(correct) / total
        print("Adv accuracy: {:.3f}".format(acc * 100))
        report.clean_train_adv_eval = acc
        return report


if __name__ == '__main__':
    path= 'models/Demo/LOGS/models/nnet_epoch_9.pt'
    model={}
    train=None
    train_dl=None
    test_dl=None
    
    
    ataque = Ataque(model, path)
    ataque.load_model(model, path)
    train_dl, test_dl = ataque.load_epoch(model)



