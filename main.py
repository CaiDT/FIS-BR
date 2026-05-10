import os
import time
import math
import torch.nn as nn
import numpy as np
import pandas as pd
import openpyxl
import torch.optim as optim
import torch.utils.data

import networks.network
from dataloader.dataloader import MainDataset as Dataset
from dataloader.dataloader import ValDataset
import argparse
import importlib
from functools import reduce
import utils
import approach

from networks import tvmodels, allmodels, set_tvmodel_head_var
from networks import network
from dataloader import transforms3d
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from TSSTANet.tsstanet import tanet, sanet, stanet, stanet_af

# params need to search (STA模型所需要的参数)
optimizer_name = 'Adam'
# lr = 0.001
frame_len = 64  # 从已处理的图像帧中采样的数量
features = 16
sigma = 0

tstart = time.time()

BATCHSIZE = 5

# Arguments
parser = argparse.ArgumentParser(description='AC-AVEC_continual')

# miscellaneous args
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU (default=%(default)s)')
parser.add_argument('--results-path', type=str, default='../results',
                    help='Results path (default=%(default)s)')
parser.add_argument('--exp-name', default=None, type=str,
                    help='Experiment name (default=%(default)s)')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed (default=%(default)s)')
parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                    help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
parser.add_argument('--save-models', action='store_true',
                    help='Save trained models (default=%(default)s)')
parser.add_argument('--last-layer-analysis', action='store_true',
                    help='Plot last layer analysis (default=%(default)s)')
parser.add_argument('--no-cudnn-deterministic', action='store_true',
                    help='Disable CUDNN deterministic (default=%(default)s)')
# dataset args
# parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
#                     help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
parser.add_argument('--num-workers', default=4, type=int, required=False,
                    help='Number of subprocesses to use for dataloader (default=%(default)s)')
parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                    help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
parser.add_argument('--batch-size', default=64, type=int, required=False,
                    help='Number of samples per batch to load (default=%(default)s)')
parser.add_argument('--num-tasks', default=4, type=int, required=False,
                    help='Number of tasks per dataset (default=%(default)s)')
# parser.add_argument('--nc-first-task', default=None, type=int, required=False,
#                     help='Number of classes of the first task (default=%(default)s)')
parser.add_argument('--use-valid-only', action='store_true',
                    help='Use validation split instead of test (default=%(default)s)')
# parser.add_argument('--stop-at-task', default=0, type=int, required=False,
#                     help='Stop training after specified task (default=%(default)s)')
# model args
parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                    help='Network architecture used (default=%(default)s)', metavar="NETWORK")
parser.add_argument('--keep-existing-head', action='store_true',
                    help='Disable removing classifier last layer (default=%(default)s)')
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained backbone (default=%(default)s)')
# training args
parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                    help='Learning approach used (default=%(default)s)', metavar="APPROACH")
parser.add_argument('--nepochs', default=200, type=int, required=False,
                    help='Number of epochs per training session (default=%(default)s)')
parser.add_argument('--lr', default=0.1, type=float, required=False,
                    help='Starting learning rate (default=%(default)s)')
parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                    help='Minimum learning rate (default=%(default)s)')
parser.add_argument('--lr-factor', default=1.5, type=float, required=False,
                    help='Learning rate decreasing factor (default=%(default)s)')
parser.add_argument('--lr-patience', default=5, type=int, required=False,
                    help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
parser.add_argument('--clipping', default=10000, type=float, required=False,
                    help='Clip gradient norm (default=%(default)s)')
parser.add_argument('--momentum', default=0.0, type=float, required=False,
                    help='Momentum factor (default=%(default)s)')
parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                    help='Weight decay (L2 penalty) (default=%(default)s)')
parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                    help='Number of warm-up epochs (default=%(default)s)')
parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                    help='Warm-up learning rate factor (default=%(default)s)')
parser.add_argument('--multi-softmax', action='store_true',
                    help='Apply separate softmax for each task (default=%(default)s)')
parser.add_argument('--fix-bn', action='store_true',
                    help='Fix batch normalization after first task (default=%(default)s)')
parser.add_argument('--eval-on-train', action='store_true',
                    help='Show train loss and accuracy (default=%(default)s)')
parser.add_argument('--br-enable', action='store_true', default=False,
                    help='Enable Balanced Replay module')
parser.add_argument('--br-memory-size', default=200, type=int,
                    help='Total memory size for Balanced Replay buffer')
parser.add_argument('--br-proto-ratio', default=0.8, type=float,
                    help='Ratio of prototype samples in BR buffer')
parser.add_argument('--br-uncert-ratio', default=0.2, type=float,
                    help='Ratio of uncertainty samples in BR buffer')
parser.add_argument('--fis-enable', action='store_true', default=False)

parser.add_argument('--fis-lambda-min', default=0.2, type=float)
parser.add_argument('--fis-lambda-max', default=0.8, type=float)

parser.add_argument('--fis-topk-ratio', default=0.3, type=float)

# gridsearch args
parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                    help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')

args, extra_args = parser.parse_known_args()
base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                   lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                   wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                   wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn,
                   eval_on_train=args.eval_on_train)  # , mu=args.mu)

GPU_id = args.gpu
DEVICE = torch.device('cuda:{}'.format(GPU_id) if torch.cuda.is_available() else 'cpu')
model_name = args.network

# Args -- Network
from networks.network import LLL_Net  # LLL_Net 为Incremental 的基本框架，可放入模型

if model_name == 'STA':

    # Generate the model.
    net = stanet_af(layers=[2, 2, 2, 2], in_channels=3, num_classes=1, k=2, features=features)
    net = torch.nn.DataParallel(net, device_ids=[GPU_id])
    init_model = net.to(DEVICE)

    network.Model = 'STA'

    # Generate the optimizers.
    # optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)

from approach.incremental_learning import Inc_Learning_Appr

Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
assert issubclass(Appr, Inc_Learning_Appr)
appr_args, extra_args = Appr.extra_parser(extra_args)
print('Approach arguments =')
for arg in np.sort(list(vars(appr_args).keys())):
    print('\t' + arg + ':', getattr(appr_args, arg))
print('=' * 108)

# Args -- Exemplars Management
from datasets.exemplars_dataset import ExemplarsDataset

Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
if Appr_ExemplarsDataset:
    assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
    appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
    print('Exemplars dataset arguments =')
    for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
        print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
    print('=' * 108)
else:
    appr_exemplars_dataset_args = argparse.Namespace()

# Args -- GridSearch
if args.gridsearch_tasks > 0:
    from gridsearch import GridSearch

    gs_args, extra_args = GridSearch.extra_parser(extra_args)
    Appr_finetuning = getattr(importlib.import_module(name='approach.finetuning'), 'Appr')
    assert issubclass(Appr_finetuning, Inc_Learning_Appr)
    GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
    print('GridSearch arguments =')
    for arg in np.sort(list(vars(gs_args).keys())):
        print('\t' + arg + ':', getattr(gs_args, arg))
    print('=' * 108)

# Network and Approach instances
utils.seed_everything(seed=args.seed)
net = LLL_Net(init_model) # LLL_Net 为Incremental 的基本框架，可放入模型
utils.seed_everything(seed=args.seed)
appr_kwargs = {**base_kwargs, **dict(**appr_args.__dict__)}

if model_name != 'STA':
    net.add_head(192, 1)
    net.to(DEVICE)

utils.seed_everything(seed=args.seed)
appr = Appr(net, DEVICE, **appr_kwargs)

# 读取 Excel 数据集
df = pd.read_excel('./datasets/avec13/label/label.xlsx')  # 根据实际路径修改

# 提取路径和标签
image_path_list = df['path'].values
label_list = df['label'].values

# 每部分的样本数量
num_tasks = 15  # 总任务数
data_per_task = 50  # 每个任务的样本数
tasks_per_section = num_tasks  # 每个部分包含的任务数
samples_per_section = tasks_per_section * data_per_task  # 每个部分总样本数

# 索引分割范围
train_start = 0
dev_start = samples_per_section
test_start = 2 * samples_per_section

# 按照新的任务数量划分数据
train_image_path_list = image_path_list[train_start:dev_start]
train_label_list = label_list[train_start:dev_start]

dev_image_path_list = image_path_list[dev_start:test_start]
dev_label_list = label_list[dev_start:test_start]

test_image_path_list = image_path_list[test_start:]
test_label_list = label_list[test_start:]


# 根据任务划分数据
def split_data_by_task(image_path_list, label_list, num_tasks, data_per_task):
    return {
        f"task{task_idx + 1}": {
            'paths': image_path_list[task_idx * data_per_task:(task_idx + 1) * data_per_task],
            'labels': label_list[task_idx * data_per_task:(task_idx + 1) * data_per_task]
        }
        for task_idx in range(num_tasks)
    }


train_data_by_task = split_data_by_task(train_image_path_list, train_label_list, num_tasks, data_per_task)
dev_data_by_task = split_data_by_task(dev_image_path_list, dev_label_list, num_tasks, data_per_task)
test_data_by_task = split_data_by_task(test_image_path_list, test_label_list, num_tasks, data_per_task)

train_transform = transforms.Compose([
    transforms3d.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms3d.RandomHorizontalFlip()
])
val_transform = None

# 创建数据加载器
def create_data_loader(data_by_task, dataset_class, transform, batch_size, shuffle, num_workers, dataset, frame_len, img_size, input_channel):
    loaders = {}
    for task_name, task_data in data_by_task.items():
        dataset_instance = dataset_class(
            img_path=task_data['paths'],
            label_value=task_data['labels'],
            dataset=dataset,
            frame_len=frame_len,
            img_size=img_size,
            input_channel=input_channel,
            transform=transform
        )
        loaders[task_name] = DataLoader(dataset_instance, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loaders
# 创建加载器
dataset = 'avec13'
train_loaders = create_data_loader(train_data_by_task, Dataset, train_transform, BATCHSIZE, shuffle=True, num_workers=8, dataset=dataset, frame_len=frame_len, img_size=224, input_channel=3)
val_loaders = create_data_loader(dev_data_by_task, ValDataset, val_transform, BATCHSIZE, shuffle=False, num_workers=0, dataset=dataset, frame_len=frame_len, img_size=224, input_channel=3)

# 访问任务的加载器
task_list = [f"task{i}" for i in range(1, args.num_tasks+1)]
train_loader_list = [train_loaders[task] for task in task_list]
val_loader_list = [val_loaders[task] for task in task_list]


taskcla = [(task_id, 4) for task_id in range(args.num_tasks)]

# GridSearch
if args.gridsearch_tasks > 0:
    ft_kwargs = {**base_kwargs, **dict(
        exemplars_dataset=GridSearch_ExemplarsDataset(transform=train_transform, class_indices=[0, 1, 2, 3]))}
    appr_ft = Appr_finetuning(net, DEVICE, **ft_kwargs)
    gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                            gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)


max_task = len(task_list)
# 初始化用于回归任务的评估指标
rmse_taw = np.zeros((max_task, max_task))
mae_taw = np.zeros((max_task, max_task))
forg_mse_taw = np.zeros((max_task, max_task))
forg_mae_taw = np.zeros((max_task, max_task))


for t, (task, ncla) in enumerate(zip(task_list, taskcla)):

    print(f"当前任务: {task}")
    print('*' * 108)
    print('Task {:2d}'.format(t+1))
    print('*' * 108)

    if 'olwf_asym' in args.approach:
        appr._task_size = ncla[1]
        appr._n_classes = ncla[1]

    # GridSearch
    if t < args.gridsearch_tasks:

        print('LR GridSearch')
        best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, train_loader_list[t], val_loader_list[t])
        # Apply to approach
        appr.lr = best_ft_lr
        gen_params = gridsearch.gs_config.get_params('general')
        for k, v in gen_params.items():
            if not isinstance(v, list):
                setattr(appr, k, v)

        # Search for best forgetting/intransigence tradeoff -- Stability Decay
        print('Trade-off GridSearch')
        best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                  t, train_loader_list[t], val_loader_list[t],
                                                                  best_ft_acc)
        # Apply to approach
        if tradeoff_name is not None:
            setattr(appr, tradeoff_name, best_tradeoff)

        print('-' * 108)

    # Train
    appr.train(t, train_loader_list[t], val_loader_list[t])
    print('-' * 108)

    # Test for regression task
    for u in range(t + 1):
        # 使用回归任务的评估损失，例如RMSE或MAE
        test_loss, rmse_taw[t, u], mae_taw[t, u] = appr.eval(u, val_loader_list[u])

        if u < t:
            # 计算当前任务的遗忘率，适用于回归任务。遗忘率可以定义为先前任务的最小误差与当前误差之间的差异。
            forg_mse_taw[t, u] = rmse_taw[t, u] - rmse_taw[t - 1, u]
            forg_mae_taw[t, u] = mae_taw[t, u] - mae_taw[t - 1, u]

        print(
            '>>> Test on task {:2d} : loss={:.3f} | TAw MAE={:.3f}, forg={:.3f} | TAw RMSE={:.3f}, forg={:.3f} <<<'.format(
                u, test_loss, rmse_taw[t, u], forg_mse_taw[t, u], mae_taw[t, u], forg_mae_taw[t, u]))


    # Print Summary
    utils.print_summary(rmse_taw, mae_taw, forg_mse_taw, forg_mae_taw)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')
