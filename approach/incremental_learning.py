import time
import torch
import numpy as np
from argparse import ArgumentParser

from torch import nn
from tqdm import tqdm

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=0.0000001, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        # return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        temp_lr = self.lr
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.lr_patience, 2)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)

            # Call scheduler step at the end of each epoch
            scheduler.step()

            clock1 = time.time()

            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0))

            # Valid
            clock3 = time.time()
            valid_loss, valid_mae, valid_mse = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, MAE={:.3f}, RMSE={:.3f} |'.format(
                clock4 - clock3, valid_loss, valid_mae, valid_mse), end='')



            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # If the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')

            print()

            # Set the final best model
            self.model.set_state_dict(best_model)



    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            targets = targets.float()
            MSE_loss = self.criterion(t, outputs, targets.to(self.device))
            MAE_loss = nn.functional.l1_loss(outputs.squeeze(), targets.to(self.device))
            mean_rmse_loss = np.sqrt(np.mean(MSE_loss.item()))
            mean_mae_loss = np.mean(MAE_loss.item())
            print(
                f"[Task {t}] train MAE loss: {mean_mae_loss:.4f} | train RMSE loss: {mean_rmse_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Backward
            self.optimizer.zero_grad()
            MSE_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):

        """Evaluation code for regression task"""
        with torch.no_grad():
            total_loss, total_mae, total_mse, total_num = 0, 0, 0, 0
            self.model.eval()
            loop = tqdm(enumerate(val_loader), total=len(val_loader))
            for batch_idx, (images, targets) in loop:
                # # Forward current model
                outputs = self.model(images.to(self.device))
                # targets = targets.to(self.device) + np.random.normal(0, targets.shape[0])
                targets = targets.to(self.device)
                loss = self.criterion(t, outputs, targets)

                # Calculate MAE and MSE
                # MAE_loss_function = self.MAE_loss_func()
                # MAE_loss = MAE_loss_function(outputs, targets.to(self.device))
                MAE_loss = torch.nn.functional.l1_loss(outputs.squeeze(), targets)
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
                        f'Step: {batch_idx + 1} | test label: {targets.squeeze().cpu().numpy()} | test predict: {outputs.squeeze().cpu().numpy()}')

            # Calculate the average loss, MAE, and MSE across all batches
            avg_loss = total_loss / total_num
            avg_mae = total_mae / total_num
            avg_mse = total_mse / total_num

            # Print final results
            # timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
            print(f'test MAE: {avg_mae:.4f}  test RMSE: {avg_mse:.4f}')

        return avg_loss, avg_mae, avg_mse

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
