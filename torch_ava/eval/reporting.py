import pandas as pd


def get_report_formatting(loss_value, acc_value, dsc_value, prec_value, rec_value):

    report_str = "loss:{:.3f} Acc.:{:.3f}" " Dice:{:.3f} Prec.:{:.3f} Recall: {:.3f}".format(
        loss_value, acc_value, dsc_value, prec_value, rec_value
    )

    return report_str
