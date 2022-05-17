from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

EPSILON = 1e-7


def compute_batch_metrics(tp, tn, fp, fn):

    prec_value = tp / (tp + fp + EPSILON)
    rec_value = tp / (tp + fn + EPSILON)

    dsc_values = 2 * (prec_value * rec_value) / (prec_value + rec_value + EPSILON)
    acc_values = (tp + tn) / (tp + tn + fn + fp)

    avg_acc = np.mean(acc_values)
    avg_dsc = np.mean(dsc_values)
    avg_prec = np.mean(prec_value)
    avg_rec = np.mean(rec_value)

    return avg_acc, avg_dsc, avg_prec, avg_rec


def compute_multiclass_occurrences(target, pred, labels_index):

    cnf_matrix = confusion_matrix(target, pred, labels=labels_index)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    return TP, TN, FP, FN
