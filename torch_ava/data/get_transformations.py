import PIL
import numpy as np
from torchvision import transforms
from monai.transforms import (
    AddChannel,
    ScaleIntensity,
    EnsureType,
)


class ToNumpy:
    def __init__(self) -> None:
        pass

    def __call__(self, input_data) -> np.array:
        return np.asarray(input_data)


class DataAugOperator:
    def __init__(self):
        self.transformations = transforms.Compose([])

    @staticmethod
    def get_Grayscale():
        return transforms.Grayscale()

    @staticmethod
    def get_RandomHorizontalFlip(p):
        return transforms.RandomHorizontalFlip(p=p)

    @staticmethod
    def get_RandomRotation(degrees, resample=PIL.Image.BILINEAR):
        return transforms.RandomRotation(degrees=degrees, resample=resample)

    @staticmethod
    def get_RandomCrop(size):
        return transforms.RandomCrop(size=size)

    @staticmethod
    def get_Normalize(std_tuple, mean_tuple):
        return transforms.Normalize(std=std_tuple, mean=mean_tuple)

    @staticmethod
    def get_ColorJitter(brightness_tuple):
        return transforms.ColorJitter(brightness=brightness_tuple)

    @staticmethod
    def get_ToNumpy():
        return ToNumpy()

    @staticmethod
    def get_AddChannel():
        return AddChannel()

    @staticmethod
    def get_ScaleIntensity():
        return ScaleIntensity()

    @staticmethod
    def get_EnsureType():
        return EnsureType()

    def set_pipeline(self, trfm_pipeline):

        for trfm_name, transform_details in trfm_pipeline.items():
            operator = getattr(self, f"get_{trfm_name}")
            data_trfrms = operator(**transform_details)
            self.transformations.transforms.append(data_trfrms)

        self.transformations.transforms.append(transforms.ToTensor())

    def get_pipeline(self):
        return self.transformations
