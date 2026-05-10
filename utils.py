import os
import torch
import random
import numpy as np

cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(rmse_taw, mae_taw, forg_mse_taw, forg_mae_taw):
    """Print summary of regression results"""
    for name, metric in zip(['MAE', 'RMSE', 'Forg MAE', 'Forg RMSE'],
                            [mae_taw, rmse_taw, forg_mae_taw, forg_mse_taw]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:6.3f} '.format(metric[i, j]), end='')  # 保留小数点后三位
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:6.3f} '.format(metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:6.3f} '.format(metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)
