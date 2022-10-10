import os
import torch
from torch.utils.tensorboard import SummaryWriter

from torch_ava.utils import create_dir_recursively
from torch_ava.eval import compute_multiclass_occurrences, compute_batch_metrics


class TensorboardLoggerOperator:
    def __init__(self, dir_path, labels_index):

        self.dir_path = dir_path
        self.dir_path = os.path.join(self.dir_path, "LOGS")
        self.dir_path = os.path.abspath(self.dir_path)
        create_dir_recursively(self.dir_path)

        self.model_dir = os.path.join(self.dir_path, "models")
        create_dir_recursively(self.model_dir)

        self.labels_index = labels_index
        self.fp, self.fn, self.tp, self.tn = 0, 0, 0, 0

        tensorboard_dir = os.path.join(self.dir_path, "tensorboard")
        self.tb_logger = SummaryWriter(tensorboard_dir)

    def log_model_configs(self, nn_obj, data, hparams):
        self.tb_logger.add_graph(nn_obj, data)

        for key, value in hparams.items():
            self.tb_logger.add_text(str(key), str(value))

    def log_scalar(self, scalar_id, scalar_value, epoch):
        self.tb_logger.add_scalar(scalar_id, scalar_value, epoch)

    def update_metrics(self, y_true, y_pred):
        countings = compute_multiclass_occurrences(y_true, y_pred, list(self.labels_index.values()))
        batch_tp, batch_tn, batch_fp, batch_fn = countings

        self.tp += batch_tp
        self.tn += batch_tn
        self.fp += batch_fp
        self.fn += batch_fn

        # Computes macro average
        avg_acc, avg_dsc, avg_prec, avg_rec = compute_batch_metrics(self.tp, self.tn, self.fp, self.fn)
        return avg_acc, avg_dsc, avg_prec, avg_rec

    def reset_metrics(self):
        self.fp, self.fn, self.tp, self.tn = 0, 0, 0, 0

    def exit(self):
        self.tb_logger.close()


class ModelOperator:
    def __init__(self, device) -> None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        cuda_devices = torch.cuda.device_count()
        cuda_codenames = [f"cuda:{idx}" for idx in range(cuda_devices)]

        if device in cuda_codenames:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

    def get_device(self):
        return self.device

    def load_model(self, dir_path, epoch):
        model_fname = "nnet_epoch_" + str(epoch) + ".pt"
        model_fpath = os.path.join(dir_path, model_fname)
        return torch.load(model_fpath)

    def save_model(self, dir_path, model, epoch):
        model_fname = "nnet_epoch_" + str(epoch) + ".pt"
        model_fpath = os.path.join(dir_path, model_fname)
        torch.save(model, model_fpath)

    def set_loss(self, loss: torch.nn.Module):
        self.loss = loss

    def set_optimizer(self, optimizer: torch.optim):
        self.optimizer = optimizer

    def compute_loss(self, y_pred, y_true):
        # criterion = torch.nn.CrossEntropyLoss()
        return self.loss(y_pred, y_true)
