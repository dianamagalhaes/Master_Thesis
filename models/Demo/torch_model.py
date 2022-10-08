import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self, inpt_dims):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.pipeline = nn.Sequential(
            self.conv1, self.bn1, self.conv2, self.bn2, self.pool, self.conv4, self.bn4, self.conv5, self.bn5,
        )

        batch_size, chanels, w, h = inpt_dims
        dummy_data = torch.ones([batch_size, chanels, w, h])
        self.img_h, self.img_w = self.pipeline(dummy_data).shape[-2], self.pipeline(dummy_data).shape[-1]

        self.fc1 = nn.Linear(24 * self.img_h * self.img_w, 6)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        print(output.shape)
        output = output.view(-1, 24 * self.img_h * self.img_w)
        print(output.shape)
        output = self.fc1(output)

        return output


# Training configs


def get_optimizer(model):

    optimizer = optim.Adadelta(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    return optimizer, scheduler


loss = torch.nn.CrossEntropyLoss()
