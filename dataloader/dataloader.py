import re
import os
import torch
import numpy as np
from PIL import Image
from math import ceil
from torch.utils.data import Dataset
from . import transforms3d

# STA项目代码
class MainDataset(Dataset):
    """将每一个受试者的视屏数据打好了包 最终返回一个 image 数据包，按策略得到的视频帧"""
    def __init__(self, img_path, label_value, dataset, frame_len=16, img_size=112, input_channel=3, transform=None):
        self.img_path = img_path
        self.label_value = label_value
        self.frame_len = frame_len
        self.img_size = img_size
        self.transform = transform
        self.input_channel = input_channel
        self.dataset = dataset

    def __len__(self):
        return len(self.label_value)

    def __getitem__(self, idx):

        label = self.label_value[idx]
        if self.dataset == 'avec13':
            image_path = os.path.join('datasets', 'avec13', self.img_path[idx])
        else:
            image_path = os.path.join('datasets', 'avec14', self.img_path[idx])

        # eg:image_path = 'datasets/avec14/Training/Northwind/243_1' 其中 Training/Northwind/243_1 为 img_path 中的内容
        # image_name 是该目录下所有图片数据的名称
        image_name = os.listdir(image_path)
        if self.dataset == 'avec14':
            image_name.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        else:
            image_name = sorted(image_name, key=str)

        if len(image_name) <= self.frame_len:
            frames = [(i * len(image_name) // self.frame_len) for i in range(self.frame_len)]
        else:
            # 1. using same sampling interval, random position
            rand_id = np.random.randint(0, len(image_name) - self.frame_len)
            frames = [i for i in range(rand_id, rand_id + self.frame_len)]

            # # 2. using full video clip, variant interval
            # frames = [(i * len(image_name) // self.frame_len) for i in range(self.frame_len)]

        image_pack = np.empty((self.frame_len, self.img_size, self.img_size, 3), np.dtype('float32'))

        for i, frame in enumerate(frames):
            image = Image.open(os.path.join(image_path, image_name[frame]))
            image = image.resize((self.img_size, self.img_size))
            image_pack[i] = np.asarray(image)

        image = np.transpose(image_pack, (3, 0, 1, 2))

        if self.transform:
            image = self.transform(image)

        if self.input_channel == 1:
            image = transforms3d.rgb_to_gray(image)
            # image = transforms3d.histeq(image)

        image = transforms3d.to_tensor(image)

        # image standardized
        image = transforms3d.normalize(image)

        return image, label
class ValDataset(Dataset):

    def __init__(self, img_path, label_value, dataset, frame_len=16, img_size=112, input_channel=3, transform=None):
        self.img_path = img_path
        self.label_value = label_value
        self.frame_len = frame_len
        self.img_size = img_size
        self.transform = transform
        self.input_channel = input_channel
        self.dataset = dataset

    def __len__(self):
        return len(self.label_value)

    def __getitem__(self, idx):

        label = self.label_value[idx]

        if self.dataset == 'avec14':
            image_path = os.path.join('datasets', 'avec14', self.img_path[idx])
        elif self.dataset == 'avec13':
            image_path = os.path.join('datasets', 'avec13', self.img_path[idx])
        elif self.dataset == 'ucf101':
            image_path = os.path.join('datasets', 'ucf101', 'ucf101_jpegs_256', self.img_path[idx])
        elif self.dataset == 'ck+':
            image_path = os.path.join('datasets', 'CK+', 'image_cropped', self.img_path[idx])
        else:
            image_path = os.path.join('datasets', 'YouTube_Faces', 'aligned_cropped', self.img_path[idx])

        image_name = os.listdir(image_path)
        if self.dataset == 'avec14':
            image_name.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        else:
            image_name = sorted(image_name, key=str)

        if len(image_name) <= self.frame_len:
            frames = []
            frames.append([(i * len(image_name) // self.frame_len) for i in range(self.frame_len)])
        else:
            # 1. using same sampling interval, random position
            rand_id = np.random.randint(0, len(image_name) - self.frame_len)
            frames = [i for i in range(rand_id, rand_id + self.frame_len)]

            # section = ceil(len(image_name) / self.frame_len)
            # slice_idx = [int(i * (self.frame_len - ((section * self.frame_len - len(image_name)) / int(section - 1))))
            #              for i in range(section)]
            # frames = []
            # for idx in slice_idx:
            #     frames.append([i for i in range(idx, idx + self.frame_len)])

        image_pack = np.empty((self.frame_len, self.img_size, self.img_size, 3), np.dtype('float32'))

        # val_image_list = []

        for i, frame in enumerate(frames):
            if type(frame) == list:
                frame = frame[0]
            val_image = Image.open(os.path.join(image_path, image_name[frame]))
            val_image = val_image.resize((self.img_size, self.img_size))
            image_pack[i] = np.asarray(val_image)

        val_image = np.transpose(image_pack, (3, 0, 1, 2))

        if self.transform:
            val_image = self.transform(val_image)

        if self.input_channel == 1:
            val_image = transforms3d.rgb_to_gray(val_image)
            # image = transforms3d.histeq(image)

        val_image = transforms3d.to_tensor(val_image)

        # image standardized
        val_image = transforms3d.normalize(val_image)

        # val_image_list.append(image)

        return val_image, label