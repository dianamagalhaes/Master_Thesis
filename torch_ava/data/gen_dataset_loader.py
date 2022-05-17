import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from torch_ava.configs import random_seed


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)


class LoaderOperator:
    def __init__(self, torch_dset, split=(0.6, 0.2, 0.2)) -> None:

        if sum(split) != 1.0:
            raise ValueError(
                "Please provide a data splitting that sums up to 1.0, to take advantage of the full dataset."
            )

        total_samples = len(torch_dset)
        idx = list(range(total_samples))

        train_ptr = int(np.floor(split[0] * total_samples))
        val_ptr = int(np.floor(split[1] * total_samples))

        np.random.seed(random_seed)
        np.random.shuffle(idx)

        self.train_idx, self.val_idx, self.test_idx = (
            idx[:train_ptr],
            idx[train_ptr : (train_ptr + val_ptr)],
            idx[(train_ptr + val_ptr) :],
        )
        print("Training samples:", train_ptr)
        print("Validation samples:", val_ptr)

    def get_loader(self, mode, torch_dset, batch_size, num_workers=2, pin_memory=True):

        if mode == "train":
            samples_idx = self.train_idx
        elif mode == "val":
            samples_idx = self.val_idx
        else:
            samples_idx = self.test_idx

        sampler = SubsetRandomSampler(samples_idx)

        data_loader = torch.utils.data.DataLoader(
            torch_dset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
        )

        return data_loader
