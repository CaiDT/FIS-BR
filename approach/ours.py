from torch.utils.data import DataLoader, ConcatDataset
from .br_buffer import BalancedReplayBuffer
from .fis_module import FISGenerator
import itertools
from datetime import time

import numpy as np
import torch
import math
from copy import deepcopy
from argparse import ArgumentParser
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .incremental_learning import Inc_Learning_Appr
# from datasets.exemplars_dataset import ExemplarsDataset
from networks.early_conv_vit_net import get_attention_list, start_rec, stop_rec, get_STA_attention_list
from networks import network
from einops import rearrange  #, reduce, repeat


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach"""


    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=0.0000001, lr_factor=3, lr_patience=5,
                 clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, sparse=False, sparse_factor=1., plast_mu=1, sym=False,
                 after_norm=False,
                 asym_penalty=0.5, perm_relu=False, jsd=False, jsd_factor=1.0, int_layer=False, pool_layers=[],
                 use_pod_factor=False,
                 reverse_relu=False, lamb=1, alpha=0.5, fi_num_samples=-1,T=2, pool_along='spatial',
                 br_memory_size=200,
                 br_proto_ratio=0.8,
                 br_uncert_ratio=0.2,
                 br_enable=True,
                 fis_enable=False,
                 fis_lambda_min=0.2,
                 fis_lambda_max=0.8,
                 fis_topk_ratio=0.3,
                 ):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)

        self.fis_enable = fis_enable

        self.fis_generator = FISGenerator(

            device=self.device,

            lambda_min=fis_lambda_min,

            lambda_max=fis_lambda_max,

            topk_ratio=fis_topk_ratio,

        )
        self.br_enable = br_enable

        self.br_buffer = BalancedReplayBuffer(

            memory_size=br_memory_size,

            proto_ratio=br_proto_ratio,

            uncertainty_ratio=br_uncert_ratio,

            device=self.device,

        )
        self.br_replay_weight = 1.0

        self.model_old = None
        self.lamb = lamb

        self.alpha = alpha
        self.num_samples = fi_num_samples
        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store fisher information weight importance
        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                           if p.requires_grad}

        self.T = T
        self.plast_mu = plast_mu
        self._task_size = 0
        self._n_classes = 0
        self._pod_spatial_factor = 3.
        self.use_pod_factor = use_pod_factor
        self.sparse_factor = sparse_factor
        self.sym = sym
        self.after_norm = after_norm
        self.asym_penalty = asym_penalty
        self.perm_relu = perm_relu
        self.jsd = jsd
        self.jsd_factor = jsd_factor
        self.sparse = sparse
        self.int_layer = int_layer
        self.pool_layers = pool_layers
        self.reverse_relu = reverse_relu
        self.pool_along = pool_along
        print(f"Using {'asymmetric' if not self.sym else 'symmetric'} version of the loss \
            function {'with permissive relu' if self.perm_relu else ''} {'after' if self.after_norm else 'before'} normalization")

    # @staticmethod
    # def exemplars_dataset_class():
    #     return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--int-layer', action='store_true', default=False, required=False,
                            help='if true, carry out pooling on the specified layers')
        parser.add_argument('--use-pod-factor', action='store_true', default=False, required=False,
                            help='Use pod factor to weigh sym/asym losses if given (default=%(default)s)')
        parser.add_argument('--reverse-relu', action='store_true', default=False, required=False,
                            help='Use relu(b-a) if given (default=%(default)s)')
        parser.add_argument('--pool-layers', default=6, type=int,
                            help='List of layers on which pooling of sym/asym loss is to be carried out(default=%(default)s)',
                            nargs='+')
        parser.add_argument('--sparse-factor', default=1., type=float, required=False,
                            help='add sparse attention regularization for asym loss')
        parser.add_argument('--sparse', action='store_true', default=False, required=False,
                            help='Use sparsity term in the loss if given (default=%(default)s)')
        parser.add_argument('--jsd', action='store_true', default=False, required=False,
                            help='Use JS divergence loss if given (default=%(default)s)')
        parser.add_argument('--jsd-factor', default=1.0, type=float, required=False,
                            help='jsd loss contribution factor')
        parser.add_argument('--sym', action='store_true', default=False, required=False,
                            help='Use symmetric version of the loss if given (default=%(default)s)')
        parser.add_argument('--after-norm', action='store_true', default=False, required=False,
                            help='Compute asymmetric loss after normalizing the attention vectors (default=%(default)s)')
        parser.add_argument('--plast_mu', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        parser.add_argument('--perm-relu', action='store_true', default=False, required=False,
                            help='Use permissive relu as choice for asymmetric loss (default=%(default)s)')
        parser.add_argument('--asym-penalty', default=0.5, type=float, required=False,
                            help='Penalty to be applied to asym if permissive relu to be the choice for asymmetric loss (default=%(default)s)')

        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')

        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--pool-along', default='spatial', required=False)
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.exemplars_dataset and len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        # return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

        return torch.optim.Adam(params, lr=self.lr)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # Add BR memory samples to train_loader
        if t > 0 and self.br_enable and not self.br_buffer.is_empty():
            trn_loader = self._make_loader_with_br(trn_loader)

        # Keep original exemplar logic if you still use exemplars_dataset
        if self.exemplars_dataset is not None:
            if len(self.exemplars_dataset) > 0 and t > 0:
                trn_loader = torch.utils.data.DataLoader(
                    ConcatDataset([trn_loader.dataset, self.exemplars_dataset]),
                    batch_size=trn_loader.batch_size,
                    shuffle=True,
                    num_workers=trn_loader.num_workers,
                    pin_memory=trn_loader.pin_memory,
                )

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        # self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        """MAS Post_train_process"""
        """Runs after training all the epochs of the task (after the train session)"""
        if self.fis_enable:

            synthetic_features = self._generate_fis_samples(
                trn_loader,
                t
            )

            if synthetic_features is not None:

                for feat in synthetic_features:
                    self.br_buffer.samples.append({
                        "image": None,
                        "target": torch.tensor(3),
                        "feature": feat.detach().cpu(),
                        "attn_score": 1.0,
                        "uncertainty": 0.0,
                        "class_id": 3,
                        "synthetic": True,
                    })

        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_importance = self.estimate_parameter_importance(trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.importance.keys():
            # Added option to accumulate importance over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
                self.importance[n] = alpha * self.importance[n] + (1 - alpha) * curr_importance[n]
            else:
                # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
                self.importance[n] = self.alpha * self.importance[n] + (1 - self.alpha) * curr_importance[n]

        if self.br_enable:
            print(f"[BR] Collecting memory candidates after task {t}...")
            br_candidates = self._collect_br_candidates(trn_loader)

            self.br_buffer.update(br_candidates)

            print(f"[BR] Memory size after update: {len(self.br_buffer)}")

            if self.fis_enable:
                print("[FIS] Generating synthetic severe samples...")

                synthetic_features, severe_target_mean = self._generate_fis_samples(
                    br_candidates
                )

                if synthetic_features is not None:
                    self.br_buffer.add_synthetic_features(
                        synthetic_features,
                        severe_target_mean
                    )

                    print(
                        f"[FIS] Added synthetic severe samples. "
                        f"Memory size: {len(self.br_buffer)}"
                    )
                else:
                    print("[FIS] No valid synthetic severe samples generated.")

    def plasticity_loss(self, old_attention_list, attention_list):


        totloss = 0.
        for i in range(len(attention_list)):
            # reshape
            p = rearrange(old_attention_list[i].view(-1, 197, 197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197, 197).to(self.device), 'b h w -> (b w) h')

            # get rid of negative values
            p = torch.abs(p)
            q = torch.abs(q)

            # transform them in probabilities
            p /= p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)

            # JS
            m = (1. / 2.) * (p + q)
            t1 = (1. / 2.) * (p * ((p / m) + 1e-05).log()).sum(dim=1)
            t2 = (1. / 2.) * (q * ((q / m) + 1e-05).log()).sum(dim=1)
            loss = t1 + t2

            # we sum the mean for each layer
            totloss += loss.mean()

        return totloss

    def calculate_permissive_relu(self, att_diff, asym_choice):
        print("Scaling..")
        relu_out_ = asym_choice(att_diff)
        penalty_factor = self.asym_penalty  #math.log(math.sqrt(
        #     self._n_classes / self._task_size
        # ))
        scaled_att_diff = torch.abs(att_diff) * penalty_factor
        # scaled_att_diff = torch.abs(att_diff) / 2.0 # make the negative values go smaller after abs() so that they are penalized less
        zero_relu_indices = relu_out_ == 0
        relu_out = relu_out_.clone()
        relu_out[zero_relu_indices] = scaled_att_diff[zero_relu_indices]
        return relu_out

    def STA_asymmetric_headwise_loss(self, old_attention_list, attention_list):
        asym_loss = "leaky_relu"
        asym_choice = torch.nn.LeakyReLU(inplace=True) if asym_loss == "leaky_relu" else \
            torch.nn.ELU(inplace=True) if asym_loss == "elu" else torch.nn.ReLU(inplace=True)

        total_spatial_loss = torch.tensor(0.).to(self.device)
        total_temporal_loss = torch.tensor(0.).to(self.device)

        # Spatial attention loss
        for old_spatial, new_spatial in zip(old_attention_list[1], attention_list[1]):
            assert old_spatial.shape == new_spatial.shape, 'Spatial attention shape mismatch'

            if self.sym:
                old_spatial = torch.pow(old_spatial, 2)
                new_spatial = torch.pow(new_spatial, 2)

            # Compute difference without reducing dimensions
            # spatial shape is (5, 2, 8, 1, 56, 56)
            spatial_diff = asym_choice(old_spatial - new_spatial)
            spatial_loss = torch.mean(torch.frobenius_norm(spatial_diff, dim=(-2, -1)))  # Compute across all spatial dimensions
            total_spatial_loss += spatial_loss

        # Temporal attention loss
        for old_temporal, new_temporal in zip(old_attention_list[0], attention_list[0]):
            assert old_temporal.shape == new_temporal.shape, 'Temporal attention shape mismatch'

            if self.sym:
                old_temporal = torch.pow(old_temporal, 2)
                new_temporal = torch.pow(new_temporal, 2)

            # Compute difference without reducing dimensions
            # temporal shape is (5, 2, 8, 64, 1, 1)
            temporal_diff = asym_choice(old_temporal - new_temporal)
            temporal_loss = torch.mean(torch.frobenius_norm(temporal_diff, dim=(-3)))  # Compute across temporal dimensions
            total_temporal_loss += temporal_loss

        totaloss = total_temporal_loss + total_spatial_loss

        return totaloss

    def asymmetric_headwise_loss(self, old_attention_list, attention_list, collapse_channels="spatial"):
        totloss = torch.tensor(0.).to(self.device)

        asym_loss = "relu"
        layers_to_pool = range(len(old_attention_list)) if not self.int_layer else [each - 1 for each in
                                                                                    self.pool_layers]

        # for i in layers_to_pool:
        # p = rearrange(old_attention_list[i].to(self.device), 's h b w -> h s b w') # rearrange to make head as the first dimension
        # q = rearrange(attention_list[i].to(self.device), 's h b w -> h s b w')

        for idx, (a, b) in enumerate(zip(old_attention_list, attention_list)):
            # each element is now of shape (96, 197, 197)
            assert a.shape == b.shape, 'Shape error'
            if self.sym:
                a = torch.pow(a, 2)
                b = torch.pow(b, 2)
            if collapse_channels == "spatial":
                a_h = a.sum(dim=2).view(a.shape[0], -1)  # [bs, w]
                b_h = b.sum(dim=2).view(b.shape[0], -1)  # [bs, w]
                a_w = a.sum(dim=3).view(a.shape[0], -1)  # [bs, h]
                b_w = b.sum(dim=3).view(b.shape[0], -1)  # [bs, h]
                a = torch.cat([a_h, a_w],
                              dim=-1)  # concatenates two [96, 197] to give [96, 394], dim = -1 does concatenation along the last axis
                b = torch.cat([b_h, b_w], dim=-1)
            elif collapse_channels == "gap":
                # compute avg pool2d over each 32x32 image to reduce the dimension to 1x1
                a = F.adaptive_avg_pool2d(a, (1, 1))[
                    ..., 0, 0]  # [..., 0, 0] preserves only the [0][0]th element of last two dimensions, i.e., [96, 197, 197] into [96], since 197x197 reduced to 1x1 and pooled together
                b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
            elif collapse_channels == "width":
                a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * h)
                b = b.sum(dim=3).view(b.shape[0], -1)
            elif collapse_channels == "height":
                a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * w)
                b = b.sum(dim=2).view(b.shape[0], -1)
            elif collapse_channels == 'pixel':
                pass

            distance_loss_weight = self.pod_spatial_factor if self.use_pod_factor else self.plast_mu
            if not self.sym:
                asym_choice = torch.nn.LeakyReLU(inplace=True) if asym_loss == "leaky_relu" else \
                    torch.nn.ELU(inplace=True) if asym_loss == "elu" else torch.nn.ReLU(inplace=True)
                if self.after_norm:
                    a = F.normalize(a, dim=1, p=2)
                    b = F.normalize(b, dim=1, p=2)

                    if self.reverse_relu:
                        diff = b - a
                    else:
                        diff = a - b
                    if self.perm_relu:
                        relu_out = self.calculate_permissive_relu(diff, asym_choice)
                    else:
                        relu_out = asym_choice(diff)
                    layer_loss = torch.mean(torch.frobenius_norm(relu_out, dim=-1)) * distance_loss_weight
                else:

                    if self.reverse_relu:
                        diff = b - a
                    else:
                        diff = a - b
                    if self.perm_relu:
                        relu_out = self.calculate_permissive_relu(diff, asym_choice)
                    else:
                        relu_out = asym_choice(diff)
                    # layer_loss = torch.mean(F.normalize(relu_out, dim=1, p=2)) # (a) good mu for this = 10
                    layer_loss = torch.mean(torch.frobenius_norm(
                        F.normalize(relu_out, dim=1, p=2))) / 100.0  # (d) works but only after /100
                    layer_loss = layer_loss * distance_loss_weight

            else:
                a = F.normalize(a, dim=1, p=2)
                b = F.normalize(b, dim=1, p=2)
                layer_loss = torch.mean(torch.frobenius_norm(a - b,
                                                             dim=-1)) * distance_loss_weight  # right now the loss is symmetric, i.e., the new model is told to attend to the same region as the old model
            totloss += layer_loss

            if self.sparse:
                attention_sparsity_term = torch.norm(torch.abs(b)) / 10.  #self.sparse_factor
                attention_sparsity_term = attention_sparsity_term * self.sparse_factor

                totloss += attention_sparsity_term

            # totloss = totloss / len(p)
        if self.sparse:
            totloss = totloss / (2 * len(layers_to_pool))
        else:
            totloss = totloss / len(layers_to_pool)
        return totloss

    def estimate_parameter_importance(self, trn_loader):
        scaling_factor = 200
        # Initialize importance matrices
        importance = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.model.named_parameters()
                      if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to accumulate L2-loss gradients
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):

            images, targets = images.to(self.device), targets.to(self.device)
            targets = targets.float()

            # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
            outputs = self.model.forward(images.to(self.device))


            loss = torch.nn.functional.mse_loss(outputs.squeeze(), targets)

            self.optimizer.zero_grad()
            loss.backward()
            # Eq. 2: accumulate the gradients over the inputs to obtain importance weights
            for n, p in self.model.model.named_parameters():
                if p.grad is not None:
                    # importance[n] += p.grad.abs() * len(targets)
                    # Incorporate both the weight and gradient into importance
                    # importance[n] += (p.grad.abs() * p.abs()) * len(targets)
                    importance[n] += scaling_factor * (p.grad.abs().detach() * p.abs().pow(2).detach()) * len(targets)
        # Eq. 2: divide by N total number of samples
        n_samples = n_samples_batches * trn_loader.batch_size
        importance = {n: (p / n_samples) for n, p in importance.items()}
        return importance

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()

        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for batch_idx, (images, targets) in enumerate(trn_loader):
            loss = 0.
            plastic_loss = 0.

            images = images.to(self.device)
            targets = targets.float()
            train_label = targets.to(self.device)

            targets_old = None

            if t > 0:
                start_rec()
                self.model_old.to(self.device)
                targets_old = self.model_old(images)
                stop_rec()

                if network.Model == 'STA':
                    old_attention_list = get_STA_attention_list()
                else:
                    old_attention_list = get_attention_list()

            start_rec()
            outputs, _ = self._forward_with_features(images)
            stop_rec()

            if network.Model == 'STA':
                attention_list = get_STA_attention_list()
            else:
                attention_list = get_attention_list()

            if t > 0:
                if network.Model == 'STA':
                    plastic_loss += self.STA_asymmetric_headwise_loss(
                        old_attention_list,
                        attention_list
                    )
                else:
                    plastic_loss += self.asymmetric_headwise_loss(
                        old_attention_list,
                        attention_list
                    )

                if self.jsd:
                    plastic_loss += self.plasticity_loss(
                        old_attention_list,
                        attention_list
                    ) * self.jsd_factor

            mse_loss, lwf_loss, loss_reg = self.criterion(
                t,
                outputs,
                train_label,
                targets_old
            )

            replay_loss = self._compute_replay_loss(images.size(0))

            MAE_loss_function = self.MAE_loss_func()
            MAE_loss = MAE_loss_function(outputs, train_label.to(self.device))

            mean_rmse_loss = np.sqrt(np.mean(mse_loss.item()))
            mean_mae_loss = np.mean(MAE_loss.item())

            print(
                f"[Task {t}] "
                f"train MAE loss: {mean_mae_loss:.4f} | "
                f"train RMSE loss: {mean_rmse_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} "
                f"Plastic:{plastic_loss:.3f} "
                f"LwF:{lwf_loss:.3f} "
                f"Mas:{loss_reg:.3f} "
                f"Replay:{replay_loss:.3f}"
            )

            loss = (
                    mse_loss
                    + plastic_loss
                    + lwf_loss
                    + loss_reg
                    + self.br_replay_weight * replay_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()


    def eval(self, t, val_loader):
        """Evaluation code for regression task"""
        with torch.no_grad():
            total_loss, total_mae, total_rmse, total_num = 0, 0, 0, 0
            self.model.eval()
            loop = tqdm(enumerate(val_loader), total=len(val_loader))
            for batch_idx, (images, targets) in loop:
                # Forward old model (if t > 0, for continual learning scenarios)
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images.to(self.device))

                outputs = self.model(images.to(self.device))

                targets = targets.to(self.device)
                rmse_loss,  lwf_loss, loss_reg  = self.criterion(t, outputs, targets, outputs_old)
                loss = rmse_loss + lwf_loss + loss_reg
                # Calculate MAE and MSE
                MAE_loss_function = self.MAE_loss_func()
                MAE_loss = MAE_loss_function(outputs, targets.to(self.device))
                mae_loss = np.mean(MAE_loss.item())
                rmse_loss = np.sqrt(np.mean(rmse_loss.item()))

                # Log the losses for this batch
                total_loss += loss.item() * len(targets)
                total_mae += mae_loss.item() * len(targets)
                total_rmse += rmse_loss.item() * len(targets)
                total_num += len(targets)

                # Print progress every 1 steps
                if (batch_idx + 1) % 1 == 0:
                    print(
                        f'Step: {batch_idx + 1} | val label: {targets.squeeze().cpu().numpy()} | val predict: {outputs.squeeze().cpu().numpy()}')

            # Calculate the average loss, MAE, and MSE across all batches
            avg_loss = total_loss / total_num
            avg_mae = total_mae / total_num
            avg_rmse = total_rmse / total_num

            # Print final results
            # timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
            print(f'val MAE: {avg_mae:.4f}  val RMSE: {avg_rmse:.4f}')

        return avg_loss, avg_mae, avg_rmse

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def MAE_loss_func(self):
        return nn.L1Loss()

    def _forward_with_features(self, images):
        """
        Forward model and return:
            outputs: task head outputs
            features: high-level embedding
        """

        outputs, features = self.model(
            images.to(self.device),
            return_features=True
        )

        return outputs, features   

    def _compute_attention_score(self, attention_list):
        """
        Compute per-sample attention score from STA attention list.

        attention_list returned by get_STA_attention_list():
            attention_list[0] = temporal attention list
            attention_list[1] = spatial attention list

        Returns:
            Tensor(B,)
        """
        if attention_list is None or len(attention_list) < 2:
            return None

        temporal_list = attention_list[0]
        spatial_list = attention_list[1]

        scores = None
        count = 0

        for att_t in temporal_list:
            # shape example: (B, k, C/k, T, 1, 1)
            s = att_t.detach().float().abs()
            s = s.view(s.size(0), -1).mean(dim=1)
            scores = s if scores is None else scores + s
            count += 1

        for att_s in spatial_list:
            # shape example: (B, k, C/k, 1, H, W)
            s = att_s.detach().float().abs()
            s = s.view(s.size(0), -1).mean(dim=1)
            scores = s if scores is None else scores + s
            count += 1

        if scores is None:
            return None

        return scores / max(count, 1)

    def _compute_uncertainty(self, outputs, targets, t):
        """
        Regression-compatible uncertainty:
            U(x) = |pred - target|
        """

        pred = outputs[t].detach().squeeze()

        y = targets.detach().float().to(pred.device).squeeze()

        if pred.dim() == 0:
            pred = pred.view(1)

        if y.dim() == 0:
            y = y.view(1)

        return torch.abs(pred - y)

    def _collect_br_candidates(self, loader, t):
        """
        Collect candidates from current task after training.
        Each candidate stores image, target, feature, attention score, uncertainty.
        """
        self.model.eval()
        candidates = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                start_rec()
                outputs, features = self._forward_with_features(images)
                stop_rec()

                if network.Model == 'STA':
                    attention_list = get_STA_attention_list()
                else:
                    attention_list = get_attention_list()

                attn_scores = self._compute_attention_score(attention_list)
                if attn_scores is None:
                    attn_scores = torch.ones(images.size(0), device=self.device)

                uncertainties = self._compute_uncertainty(outputs, targets, t)

                for i in range(images.size(0)):
                    class_id = self.br_buffer._to_class_id(targets[i])

                    candidates.append({
                        "image": images[i].detach().cpu(),
                        "target": targets[i].detach().cpu(),
                        "feature": features[i].detach().cpu(),
                        "attn_score": float(attn_scores[i].detach().cpu()),
                        "uncertainty": float(uncertainties[i].detach().cpu()),
                        "class_id": class_id,
                    })

        return candidates

    def _make_loader_with_br(self, trn_loader):
        """
        Combine current task dataset with BR memory dataset.
        """
        if not self.br_enable or self.br_buffer.is_empty():
            return trn_loader

        br_dataset = self.br_buffer.get_dataset()
        merged_dataset = ConcatDataset([trn_loader.dataset, br_dataset])

        return DataLoader(
            merged_dataset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )

    def _collect_features_for_fis(self,loader,t):
        self.model.eval()
        severe_features = []
        majority_features = []
        for images, targets in loader:

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs, features = self._forward_with_features(images)

            outputs_t = outputs[t]

            loss = F.mse_loss(
                outputs_t.squeeze(),
                targets.float()
            )

            self.optimizer.zero_grad()

            loss.backward(retain_graph=True)

            gradients = features.grad

            importance = self.fis_generator.compute_importance(
                features.detach(),
                gradients.detach()
            )

            for i in range(features.size(0)):

                class_id = self.br_buffer._to_class_id(
                    targets[i]
                )

                if class_id == 3:
                    severe_features.append(features[i].detach())
                else:
                    majority_features.append(features[i].detach())

        severe_features = torch.stack(severe_features)
        majority_features = torch.stack(majority_features)

        return severe_features, majority_features, importance

    def _generate_fis_samples(self,loader,t):

        severe_features, majority_features,importance = self._collect_features_for_fis(
            loader,
            t
        )

        severe_prototype = severe_features.mean(dim=0)

        synthetic = self.fis_generator.generate(
            severe_features=severe_features,
            majority_features=majority_features,
            importance_scores=importance,
            severe_prototype=severe_prototype,
            num_generate=50
        )

        return synthetic


    def _forward_with_features(self, images):
        return self.model(images.to(self.device), return_features=True)

    def _predict_from_features(self, features):
        """
        STA regression head is inside self.model.model.fc.
        Output range is mapped to BDI-II score [0, 63].
        """
        out = self.model.model.fc(features)
        out = out * 63
        out = out.view(out.size(0))
        return out

    def _compute_attention_score(self, attention_list):
        if attention_list is None or len(attention_list) < 2:
            return None

        temporal_list = attention_list[0]
        spatial_list = attention_list[1]

        score = None
        count = 0

        for att in temporal_list:
            s = att.detach().float().abs()
            s = s.view(s.size(0), -1).mean(dim=1)
            score = s if score is None else score + s
            count += 1

        for att in spatial_list:
            s = att.detach().float().abs()
            s = s.view(s.size(0), -1).mean(dim=1)
            score = s if score is None else score + s
            count += 1

        if score is None:
            return None

        return score / max(count, 1)

    def _compute_uncertainty(self, outputs, targets):
        pred = outputs.detach().view(-1)
        y = targets.detach().float().to(pred.device).view(-1)
        return torch.abs(pred - y)

    def _collect_br_candidates(self, loader):
        self.model.eval()
        candidates = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device).float()

                start_rec()
                outputs, features = self._forward_with_features(images)
                stop_rec()

                if network.Model == 'STA':
                    attention_list = get_STA_attention_list()
                else:
                    attention_list = get_attention_list()

                attn_scores = self._compute_attention_score(attention_list)

                if attn_scores is None:
                    attn_scores = torch.ones(images.size(0), device=self.device)

                uncertainties = self._compute_uncertainty(outputs, targets)

                for i in range(images.size(0)):
                    candidates.append({
                        "image": images[i].detach().cpu(),
                        "target": targets[i].detach().cpu(),
                        "feature": features[i].detach().cpu(),
                        "attn_score": float(attn_scores[i].detach().cpu()),
                        "uncertainty": float(uncertainties[i].detach().cpu()),
                        "class_id": self.br_buffer.bdi_to_class(targets[i]),
                        "synthetic": False,
                    })

        return candidates

    def _collect_features_for_fis(self, candidates):
        severe_features = []
        severe_targets = []
        majority_features = []

        for item in candidates:
            if int(item["class_id"]) == 3:
                severe_features.append(item["feature"])
                severe_targets.append(item["target"])
            else:
                majority_features.append(item["feature"])

        if len(severe_features) == 0 or len(majority_features) == 0:
            return None, None, None

        severe_features = torch.stack(severe_features, dim=0)
        majority_features = torch.stack(majority_features, dim=0)
        severe_targets = torch.stack(severe_targets).float()

        severe_target_mean = severe_targets.mean().item()

        return severe_features, majority_features, severe_target_mean

    def _generate_fis_samples(self, candidates):
        severe_features, majority_features, severe_target_mean = self._collect_features_for_fis(candidates)

        if severe_features is None:
            return None, None

        importance = self.fis_generator.get_feature_importance_from_fc(
            self.model.model.fc
        )

        synthetic_features = self.fis_generator.generate(
            severe_features=severe_features,
            majority_features=majority_features,
            importance=importance,
            num_generate=50,
        )

        return synthetic_features, severe_target_mean

    def _compute_replay_loss(self, batch_size):
        if not self.br_enable or self.br_buffer.is_empty():
            return torch.tensor(0., device=self.device)

        replay = self.br_buffer.sample(batch_size)

        if replay is None:
            return torch.tensor(0., device=self.device)

        replay_loss = torch.tensor(0., device=self.device)

        if "real_images" in replay:
            real_images = replay["real_images"].to(self.device)
            real_targets = replay["real_targets"].to(self.device)

            real_outputs = self.model(real_images)
            replay_loss = replay_loss + F.mse_loss(
                real_outputs.view(-1),
                real_targets.view(-1)
            )

        if "syn_features" in replay:
            syn_features = replay["syn_features"].to(self.device)
            syn_targets = replay["syn_targets"].to(self.device)

            syn_outputs = self._predict_from_features(syn_features)

            replay_loss = replay_loss + F.mse_loss(
                syn_outputs.view(-1),
                syn_targets.view(-1)
            )

        return replay_loss




    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        MSE_loss_func = nn.MSELoss()
        lwf_loss = 0
        loss_reg = 0
        outputs = outputs.squeeze()


        if t > 0:
            # Knowledge distillation loss for all previous tasks
            # lamb * Loss_Lwf
            outputs_old = outputs_old.squeeze()
            lwf_loss = self.lamb * MSE_loss_func(outputs, outputs_old)

            # Eq. 3: memory aware synapses regularizer penalty
            asym_loss = "relu"
            asym_choice = torch.nn.LeakyReLU(inplace=True) if asym_loss == "leaky_relu" else \
                torch.nn.ELU(inplace=True) if asym_loss == "elu" else torch.nn.ReLU(inplace=True)
            for n, p in self.model.model.named_parameters():
                if n in self.importance.keys():
                    # loss_reg += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2)) / 2
                    loss_reg += torch.sum(self.importance[n] * asym_choice(p.abs() - self.older_params[n].abs()).pow(2)) / 2



        mse_loss = MSE_loss_func(outputs, targets)

        return mse_loss, self.lamb * lwf_loss, self.lamb * loss_reg
