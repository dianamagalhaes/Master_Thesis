import numpy as np
import os
import pydicom as pyd
from torch.utils.data import Dataset

# Custom libraries
from torch_ava.third_party import ccab_loader as database_loader


class CCAB_Dataset(Dataset):

    cmr_protocol = "cine_SA"

    def __init__(self, cardiomr_base_dir: str, dataframe_path: str, transform=None):

        self.dset_base_dir = cardiomr_base_dir
        self.transform = transform

        dataframe = database_loader.load_patients_dataframe(dataframe_path)

        case_cmr_data = dataframe[CCAB_Dataset.cmr_protocol].tolist()
        case_cmr_data = [frame_path for case in case_cmr_data for sa_slice in case for frame_path in sa_slice]

        self.dataset = case_cmr_data
        slice_type = dataframe["slice_type"].values

        self.y = np.concatenate([np.array(slices_class_list).ravel() for slices_class_list in slice_type])

        del dataframe, case_cmr_data

        self.classes = np.unique(self.y)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):

        x = pyd.dcmread(f"{os.path.join(self.dset_base_dir, self.dataset[index])}.dcm").pixel_array
        x = x[:, :, np.newaxis].astype(np.float32)

        if self.transform:
            x = self.transform(x)

        y = self.class_to_idx[self.y[index]]

        return x, y


if __name__ == "__main__":

    import platform

    if platform.node() == "nea138-lt":
        cardiomr_base_dir = "/home/apinto/Documents/repos/cardiomr_dl"
        dataframe_path = "/home/apinto/Documents/projects/msc_thesis_DianaMag/models/SA_Classification_AI4MED/EXTRA_DETAILS/test_dataframe.csv"

    ccab_dset = CCAB_Dataset(cardiomr_base_dir=cardiomr_base_dir, dataframe_path=dataframe_path)
    print("Numpy image array", ccab_dset.__getitem__(0)[0], "\n Class Index", ccab_dset.__getitem__(0)[1])
