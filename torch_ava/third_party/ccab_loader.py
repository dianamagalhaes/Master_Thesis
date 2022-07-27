
import os
import pandas as pd
import random

# Custom Libraries
from cardiomr.dicom import dcm_tools
from cardiomr.utils import str_tools


def load_patients_dataframe(path_to_file, sample_percentage=1):
    """Loads patients info in the form of a pandas dataframe.

    Args:
        path_to_file (str): path to the csv file with the dataframe.
        sample_percentage (float): random sampling to be done while loading the dataframe.
            Defaults to 1.

    Raises:
        ValueError: File with dataframe must be created before.

    Returns:
        pd.Dataframe: pandas dataframe with all patients info.

    """
    if not os.path.exists(path_to_file):
        raise ValueError("File with dataframe does not exist. Create it or do dvc pull.")

    skiprows_f = None
    if 0 < sample_percentage < 1:
        skiprows_f = lambda i: i > 0 and random.random() > sample_percentage

    conversion_to_list = lambda x: str_tools.convert_string_representation_to_object(x)
    metadata_conversion = lambda x: dcm_tools.convert_metadata(str_tools.convert_string_representation_to_object(x))

    conv = {
        "cine_SA": conversion_to_list,
        "cine_2ch": conversion_to_list,
        "cine_3ch": conversion_to_list,
        "cine_4ch": conversion_to_list,
        "bullseye": conversion_to_list,
        "insertion_points": conversion_to_list,
        "slice_type": conversion_to_list,
        "dim_reduction": conversion_to_list,
        "cine_SA_metadata": metadata_conversion,
        "cine_2ch_metadata": metadata_conversion,
        "cine_3ch_metadata": metadata_conversion,
        "cine_4ch_metadata": metadata_conversion,
        "biometric_data": metadata_conversion,
        "left_ventricle_volumes": conversion_to_list,
        "right_ventricle_volumes": conversion_to_list,
        "lv_sa_horos_measurements": conversion_to_list,
        "rv_sa_horos_measurements": conversion_to_list,
        "lv_sa_measurements": conversion_to_list,
        "rv_sa_measurements": conversion_to_list,
        "flow_magnitude": conversion_to_list,
        "flow_phase": conversion_to_list,
        "flow": conversion_to_list,
        "vessels_annotation": conversion_to_list,
        "lge_SA": conversion_to_list,
        "lge_2ch": conversion_to_list,
        "lge_3ch": conversion_to_list,
        "lge_4ch": conversion_to_list,
        "lge_SA_metadata": conversion_to_list,
        "lge_2ch_metadata": conversion_to_list,
        "lge_3ch_metadata": conversion_to_list,
        "lge_4ch_metadata": conversion_to_list,
        "cine_SA_vtk": conversion_to_list,
        "lmks_obs1": conversion_to_list,
        "lmks_obs2": conversion_to_list,
    }

    patients_dataframe = pd.read_csv(path_to_file, converters=conv, skiprows=skiprows_f)

    return patients_dataframe
