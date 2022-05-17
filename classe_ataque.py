import os
import torch
import tensorflow as tf
from tensorflow import keras

model = torch.jit.load('model_scripted.pt')
print(model)
