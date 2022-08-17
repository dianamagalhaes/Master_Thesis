import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from monai.networks.nets import DenseNet121


# Training configs
def get_optimizer(model):

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    return optimizer, scheduler


loss = torch.nn.CrossEntropyLoss()

model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6)

