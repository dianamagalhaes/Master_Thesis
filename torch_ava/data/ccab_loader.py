import numpy as np
import os
import pydicom as pyd
from torch.utils.data import Dataset

from cardiomr.utils import image_processor

# Custom libraries
from torch_ava.third_party import ccab_loader as database_loader

def filter_frame_types_by_slice_phase(patients_frame_types, patients_diastole, patients_systole, slice_phase="both"):
    """Filters every patient slice in the patients_frame_types considering the slice phase.

    Args:
        patients_frame_types (list): list of lists with patients frame types.
        patients_diastole (list): list with patients diastole times.
        patients_systole (list): list with patients systole times.
        slice_phase (str, optional): flag to filter the patients frame types. Options: {"diastole", "systole", "both",
            "all"}. If "all" no filtering is performed, if "both" then the predictions are filtered for both the
            systole and diastole (2 frames per slice). Defaults to "both".

    Returns:
        list: list of lists with patients frame types filtered according to the slice_phase.

    """
    if slice_phase == "all":
        return patients_frame_types

    filtered_patients_frame_types = []

    for patient_frame_types, diastole_time, systole_time in zip(
        patients_frame_types, patients_diastole, patients_systole
    ):

        filtered_patient_slices_frame_types = []
        diastole_time = int(diastole_time)
        systole_time = int(systole_time)

        for patient_slice in patient_frame_types:
            filtered_patient_slice = []

            if slice_phase in ("diastole", "both"):
                filtered_patient_slice.append(patient_slice[diastole_time])
            if slice_phase in ("systole", "both"):
                filtered_patient_slice.append(patient_slice[systole_time])

            filtered_patient_slices_frame_types.append(filtered_patient_slice)

        filtered_patients_frame_types.append(filtered_patient_slices_frame_types)

    return filtered_patients_frame_types

class CCAB_Dataset(Dataset):

    cmr_protocol = "cine_SA"
    FRAME_TYPES = ["top", "basal", "medial", "apical", "bottom"]

    def __init__(self, cardiomr_base_dir: str, dataframe_path: str):

        self.dset_base_dir = cardiomr_base_dir

        dataframe = database_loader.load_patients_dataframe(dataframe_path)

        cases_diastole = dataframe["diastole_time"].to_list()
        cases_systole = dataframe["systole_time"].to_list()

        # --- Imaging paths
        cases_cmr_img_paths = dataframe[CCAB_Dataset.cmr_protocol].tolist()
        cases_cmr_img_paths  = filter_frame_types_by_slice_phase(
                cases_cmr_img_paths, cases_diastole, cases_systole, slice_phase="systole"
            )
        cases_cmr_img_paths = [frame_path for case in cases_cmr_img_paths for sa_slice in case for frame_path in sa_slice]
        
        # --- Slice Labels
        cases_gt = dataframe["slice_type"].to_list()
        cases_gt = filter_frame_types_by_slice_phase(
                cases_gt, cases_diastole, cases_systole, slice_phase="systole"
            )
        cases_gt = [slice_label for case in cases_gt for sa_slice in case for slice_label in sa_slice]

        self.dataset = cases_cmr_img_paths
        self.y = cases_gt

        # self.y = np.concatenate([np.array(slices_class_list).ravel() for slices_class_list in slice_type])

        del dataframe, cases_cmr_img_paths, cases_gt

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(CCAB_Dataset.FRAME_TYPES)}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):

        x = pyd.dcmread(f"{os.path.join(self.dset_base_dir, self.dataset[index])}.dcm").pixel_array.astype(np.int32)
        # x = image_processor.crop_image_with_size(x, (192, 160))

        if x.shape != (192, 156):
            if x.shape == (156, 192):
                x = np.transpose(x, (1, 0))
            
        y = self.class_to_idx[self.y[index]]
        x = x[:, :, np.newaxis]
        return x, y

