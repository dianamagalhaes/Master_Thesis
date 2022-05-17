import torch
from tqdm import tqdm

from torch_ava.eval import get_report_formatting


class Trainer:

    tqdm_bar_format = "{l_bar}{bar}| {n:.0f}/{total_fmt} || ETA {remaining} ({elapsed}){postfix}"

    def __init__(self, model_operator: object, tb_logger: object, epochs: int) -> None:
        self.model_operator = model_operator
        self.tb_logger = tb_logger
        self.epochs = epochs

    def train(self, model, data_loader, epoch):

        prog_bar = tqdm(data_loader, unit="batch", bar_format=Trainer.tqdm_bar_format)
        prog_bar.set_description("Train:: EP. {}/{}".format(epoch, self.epochs))

        model.train()

        train_loss, samples, corrects = 0, 0, 0
        self.tb_logger.reset_metrics()

        for iterations, (data, target) in enumerate(prog_bar):
            data = data.to(self.model_operator.get_device())
            target = target.to(self.model_operator.get_device())

            self.model_operator.optimizer.zero_grad()
            out = model(data)

            batch_loss = self.model_operator.compute_loss(out, target)
            batch_loss.backward()
            self.model_operator.optimizer.step()
            # sum up batch loss
            train_loss += batch_loss.item()

            avg_loss = train_loss / (iterations + 1)

            # get the index of the max log-probability
            pred = out.argmax(dim=1, keepdim=True)
            corrects += pred.eq(target.view_as(pred)).sum().item()
            samples += len(data)

            metrics = self.tb_logger.update_metrics(target.cpu(), pred.cpu())

            report_str = get_report_formatting(
                loss_value=avg_loss,
                acc_value=corrects / samples,
                dsc_value=metrics[1],
                prec_value=metrics[2],
                rec_value=metrics[3],
            )

            prog_bar.set_postfix_str(report_str)

        self.tb_logger.log_scalar("learning_rate", self.model_operator.optimizer.param_groups[0]["lr"], epoch)
        self.tb_logger.log_scalar("Loss/train", avg_loss, epoch)
        self.tb_logger.log_scalar("Accuracy/train", corrects / samples, epoch)
        self.tb_logger.log_scalar("Dice Score/train", metrics[1], epoch)
        self.tb_logger.log_scalar("Precision/train", metrics[2], epoch)
        self.tb_logger.log_scalar("Recall/train", metrics[3], epoch)

        self.model_operator.save_model(self.tb_logger.model_dir, model, epoch)

    def validation(self, model, data_loader, epoch):

        prog_bar = tqdm(data_loader, unit="batch", bar_format=Trainer.tqdm_bar_format)
        prog_bar.set_description("Validation:: EP. {}/{}".format(epoch, self.epochs))

        # Note! When the validation data_loader comes from the same Pytorch Data Loader as in the training data loader
        # len(val_loader.dataset) will give the totality of the dataset!!!

        model.eval()

        self.tb_logger.reset_metrics()
        val_loss, samples, corrects = 0, 0, 0

        with torch.no_grad():

            for iterations, (data, target) in enumerate(prog_bar):

                data = data.to(self.model_operator.get_device())
                target = target.to(self.model_operator.get_device())

                output = model(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)

                batch_val_loss = self.model_operator.compute_loss(output, target).item()

                # sum up batch loss
                val_loss += batch_val_loss
                avg_loss = val_loss / (iterations + 1)

                corrects += pred.eq(target.view_as(pred)).sum().item()
                samples += len(data)

                metrics = self.tb_logger.update_metrics(target.cpu(), pred.cpu())

                report_str = get_report_formatting(
                    loss_value=avg_loss,
                    acc_value=corrects / samples,
                    dsc_value=metrics[1],
                    prec_value=metrics[2],
                    rec_value=metrics[3],
                )

                np_preds = pred.detach().cpu().numpy()
                np_preds = np_preds.flatten().tolist()
                np_targets = target.detach().cpu().numpy()
                np_targets = np_targets.tolist()

                prog_bar.set_postfix_str(report_str)

            self.tb_logger.log_scalar("Loss/val", avg_loss, epoch)
            self.tb_logger.log_scalar("Accuracy/val", corrects / samples, epoch)
            self.tb_logger.log_scalar("Dice Score/val", metrics[1], epoch)
            self.tb_logger.log_scalar("Precision/val", metrics[2], epoch)
            self.tb_logger.log_scalar("Recall/val", metrics[3], epoch)

    def run_epochs(self, model, train_data_loader, val_data_loader, scheduler):
        device = self.model_operator.get_device()
        model.to(device)
        for epoch in range(1, self.epochs + 1):
            self.train(model, train_data_loader, epoch)
            self.validation(model, val_data_loader, epoch)
            if scheduler is not None:
                scheduler.step()
