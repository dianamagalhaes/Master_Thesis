import os
import numpy as np


def load_training_params(params_path):

    if os.path.isfile(params_path):
        train_mean = []
        train_std_dev = []
        mean_std_array = np.loadtxt(params_path, comments="#")

        for pair in mean_std_array:
            train_mean.append(pair[0])
            train_std_dev.append(pair[1])
        print("Loading mean and standard deviation from file")

        return train_mean, train_std_dev

    else:
        raise ValueError("Please provide a valid path", params_path)


def spatial_bayesian_postproc(patient_frame_types, mean, std_deviation):
    """Post process a single patient using the naive bayes method.

    In this method the predictions made by the model constitute the likehood. The mean and standard deviation are used
    to compute the prior probabilities along with normalization. The prior probabilities are multiplied by the likehood
    of every frame for a given patient in order to obtain posterior probabilities, which are the post processed frame
    types.

    Args:
        patient_frame_types (list): the predictions of a single patient.
        mean (list): list with the mean values, one for each class
        std_deviation (list): list with the standard deviation values, one for each class

    Returns:
        list: list of post processed predictions obtained through a spatial bayesian method

    """
    n_slices = len(patient_frame_types)
    normalized_slices = []
    normalized_prob = []
    processed_patient_frame_types = []

    # get prior probabilities for patient slices
    for i in range(n_slices):
        normalized_slices.append(i / (n_slices - 1))
        not_normalized_prob = []
        for class_mean, class_stdev in zip(mean, std_deviation):
            not_normalized_prob.append(gaussian_function(normalized_slices[i], class_mean, class_stdev))
        normalized_prob.append(normalize_slice_probability(not_normalized_prob))

    for slice_number, slice_values in enumerate(patient_frame_types):
        processed_slices = []
        for frame_values in slice_values:
            post_processed_pred = []
            for class_frame_value, class_norm_prob in zip(frame_values, normalized_prob[slice_number]):
                post_processed_class_pred = class_frame_value * class_norm_prob
                post_processed_pred.append(post_processed_class_pred)
            processed_slices.append(post_processed_pred)
        processed_patient_frame_types.append(processed_slices)

    return processed_patient_frame_types
