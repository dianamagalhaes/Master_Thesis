import pandas as pd
import torch

from torch_ava.eval import gen_pred_report


class Evaluator:
    def __init__(self, data_loader, model_operator, xlsx_fname="Results.xlsx") -> None:
        self.model_operator = model_operator
        self.data_loader = data_loader

        self.xlsx_fname = xlsx_fname

    def test(self, epoch):

        model = self.model_operator.load_model(epoch)

        model.eval()
        result_dataframe = pd.DataFrame()

        with torch.no_grad():

            for iterations, (data, target) in enumerate(self.data_loader):

                data = data.to(self.model_operator.get_device())
                target = target.to(self.model_operator.get_device())

                output = model(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)

                np_preds = pred.detach().cpu().numpy().flatten().tolist()
                np_targets = target.numpy().tolist()

                result_dataframe = result_dataframe.append(
                    [{"Batch_ID": iterations, "y_pred": np_preds, "y_true": np_targets}], ignore_index=True
                )

        sheet_name = f"EP-{epoch}"
        result_dataframe = result_dataframe.set_index(["Batch_ID"]).apply(pd.Series.explode).reset_index()

        gen_pred_report(result_dataframe, fname=self.xlsx_fname, sheet_name=sheet_name)
