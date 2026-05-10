import numpy as np
import torch
from copy import deepcopy
from argparse import ArgumentParser

from torch import nn
from tqdm import tqdm

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=1, T=2):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)


    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            targets = targets.float()
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images.to(self.device))
            # Forward current model
            outputs = self.model(images.to(self.device))
            mse_loss, lwf_loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
            loss = mse_loss + lwf_loss
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_mae, total_mse, total_num = 0, 0, 0, 0
            self.model.eval()
            loop = tqdm(enumerate(val_loader), total=len(val_loader))
            for batch_idx, (images, targets) in loop:
                # Forward old model (if t > 0, for continual learning scenarios)
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images.to(self.device))

                # Forward current model
                outputs = self.model(images.to(self.device))
                mse_loss, lwf_loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                loss = mse_loss + lwf_loss

                # Calculate MAE and MSE
                MAE_loss_function = nn.L1Loss()
                MAE_loss = MAE_loss_function(outputs, targets.to(self.device))
                mae_loss = np.mean(MAE_loss.item())
                mse_loss = np.sqrt(np.mean(loss.item()))

                # Log the losses for this batch
                total_loss += loss.item() * len(targets)
                total_mae += mae_loss.item() * len(targets)
                total_mse += mse_loss.item() * len(targets)
                total_num += len(targets)

                # Print progress every 1 steps
                if (batch_idx + 1) % 1 == 0:
                    print(
                        f'Step: {batch_idx + 1} | val label: {targets.squeeze().cpu().numpy()} | val predict: {outputs.squeeze().cpu().numpy()}')

            # Calculate the average loss, MAE, and MSE across all batches
            avg_loss = total_loss / total_num
            avg_mae = total_mae / total_num
            avg_mse = total_mse / total_num

            # Print final results
            # timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
            print(f'val MAE: {avg_mae:.4f}  val MSE: {avg_mse:.4f}')

        return avg_loss, avg_mae, avg_mse

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        MSE_loss_func = nn.MSELoss()
        lwf_loss = 0
        outputs = outputs.squeeze()

        # Knowledge distillation loss for all previous tasks (LwF)
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            # lamb * Loss_Lwf
            outputs_old = outputs_old.squeeze()
            lwf_loss = self.lamb * MSE_loss_func(outputs, outputs_old)


        mse_loss = MSE_loss_func(outputs, targets)

        return mse_loss, lwf_loss