import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl import load_workbook


def gen_pred_report(result_dataframe, fname="Results.xlsx", sheet_name="Test_1"):
    add_wb = False
    if os.path.exists(fname):
        book = load_workbook(fname)
        add_wb = True

    writer = pd.ExcelWriter(fname)

    if add_wb:
        writer.book = book

    # Data to add
    result_dataframe.to_excel(writer, index=False, sheet_name=sheet_name)

    writer.save()
    writer.close()


def load_report(fname="Results.xlsx", sheet_name="Test_1"):
    xls = pd.ExcelFile(fname)
    results = pd.read_excel(xls, sheet_name)

    y_pred = results["y_pred"].tolist()
    y_true = results["y_true"].tolist()

    return y_pred, y_true


def get_report_formatting(loss_value, acc_value, dsc_value, prec_value, rec_value):

    report_str = "loss:{:.3f} Acc.:{:.3f}" " Dice:{:.3f} Prec.:{:.3f} Recall: {:.3f}".format(
        loss_value, acc_value, dsc_value, prec_value, rec_value
    )

    return report_str
