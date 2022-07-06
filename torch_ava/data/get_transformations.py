import PIL
from torchvision import transforms


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

    def set_pipeline(self, trfm_pipeline):

        for trfm_name, transform_details in trfm_pipeline.items():
            operator = getattr(self, f"get_{trfm_name}")
            data_trfrms = operator(**transform_details)
            self.transformations.transforms.append(data_trfrms)

        self.transformations.transforms.append(transforms.ToTensor())

    def get_pipeline(self):
        return self.transformations
