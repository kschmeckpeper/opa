# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import glob
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import PIL
import argparse
import ast

import random

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)

class ShapeStacksDataset(Dataset):
    def __init__(self,
                 sequence_length=16,
                 context_length=1,
                 radius=35,
                 mod='rgb',
                 dataset='ss3',
                 base_path='data/shapestacks/',
                 normalize_images=False,
                 is_train=False,
                 rand_start=False):
        super().__init__()

        self.is_train = is_train
        self.rand_start = rand_start
        fov = 35
        print("base path:", base_path)
        common_list = os.path.join(base_path, 'splits')
        image_dir = os.path.join(base_path, 'frc_%d' % fov)
        if dataset in ['ss3']:
            common_list = os.path.join(common_list, 'env_ccs+blocks-hard+easy-h=3-vcom=1+2+3-vpsf=0')
        else:
            num = int(dataset[2])
            common_list = os.path.join(common_list, 'env_ccs+blocks-hard+easy-h=%d-vcom=1+2+3+4+5+6-vpsf=0' % num)

        print("common_list:", common_list)
        if is_train:
            list_path = os.path.join(common_list, 'train.txt')
        else:
            list_path = os.path.join(common_list, 'eval.txt')

        if not os.path.exists(list_path):
            print('not exists', list_path)
            raise FileExistsError

        self.RW = self.RH = 112 
        self.W = self.H = 112

        self.orig_W = self.orig_H = 224
        self.box_rad = radius

        self.image_dir = image_dir
        self.ext = '.jpg'
        self.sequence_length = sequence_length
        self.context_length = context_length
        self.num_obj = 0
        self.training = is_train
        self.modality = mod

        transform = [Resize((self.H, self.W)), T.ToTensor()]
        obj_transform = [Resize((self.RH, self.RW)), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
            obj_transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.obj_transform = T.Compose(obj_transform)

        with open(list_path) as fp:
            self.index_list = [line.split()[0] for line in fp]
        self.roidb = self.parse_gt_roidb()
        eg_path = glob.glob(os.path.join(self.image_dir, self.index_list[0], self.modality + '*' + self.ext))[0]
        self.image_pref = '-'.join(os.path.basename(eg_path).split('-')[0:-1])

    def parse_gt_roidb(self):
        roidb = {}
        for index in self.index_list:
            
            original_boxes = True #False #True
            
            if original_boxes:
                gt_path = os.path.join(self.image_dir, index, 'cam_1.npy')
                bbox = np.load(gt_path) ## 32, 3, 2 in (0, 224) coor
                bbox_with_sizes = np.ones((bbox.shape[0], bbox.shape[1], 4))
                bbox_with_sizes[:, :, :2] = bbox
                bbox_with_sizes[:, :, 2:] *= 2*self.box_rad / self.orig_W
                roidb[index] = bbox_with_sizes # 32 x 6 x 4
                self.num_obj = bbox.shape[1]
            else:
                gt_path = os.path.join(self.image_dir, index, 'cam_1_maskrcnn.npy')
                bboxes = np.load(gt_path, allow_pickle=True) # 32-dim list, each locations is an N x 4 numpy array (N varies based on the image)
                roidb[index] = bboxes
        return roidb

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        """
        :return: src, dst. each is a list of object
        - 'image': FloatTensor of shape (dt, C, H, W). resize and normalize to faster-rcnn
        - 'crop': (O, C, RH, RW) RH >= 224
        - 'bbox': (O, 4) in xyxy (0-1) / xy logw logh
        - 'trip': (T, 3)
        - 'index': (dt,)
        """
        images = []
        all_crops = []
        norm_bboxes = []

        if self.is_train and self.rand_start:
            start_t = random.randint(0, 30-2*(self.sequence_length + self.context_length))
        else:
            start_t = 4
        for dt_2 in range(self.sequence_length + self.context_length):
            dt = start_t+dt_2 * 2
            this_index = self.get_index_after(self.index_list[index], dt)
            norm_bbox = self.roidb[self.index_list[index]][dt] # (O, 2)
            bboxes = np.vstack((norm_bbox[:, 0] * self.orig_W, norm_bbox[:, 1] * self.orig_H)).T
            image, crops = self._read_image(this_index, bboxes)

            images.append(image)
            all_crops.append(crops)
            norm_bboxes.append(torch.FloatTensor(norm_bbox))

        data = {
                'context_frames': images[:self.context_length],
                'target_frames': images[self.context_length:],
                'context_crops': all_crops[:self.context_length],
                'target_crops': all_crops[self.context_length:],
                'context_norm_bboxes': norm_bboxes[:self.context_length],
                'target_norm_bboxes': norm_bboxes[self.context_length:],
                }
        
        for k in data:
            data[k] = torch.stack(data[k])
     
        data['context_actions'] = torch.zeros(self.context_length, 2)
        data['actions'] = torch.zeros(self.sequence_length, 2)
        return data

    def get_index_after(self, index, dt):
        return os.path.join(index, self.image_pref + '-%02d' % dt)

    def _read_image(self, index, bboxes):
        image_path = os.path.join(self.image_dir, index) + self.ext
        with open(image_path, 'rb') as f:
            with Image.open(f) as image:
                dst_image = self.transform(image.convert('RGB'))
                crops = self._crop_image(index, image, bboxes)
        return dst_image, crops

    def _crop_image(self, index, image, box_center):
        crop_obj = []
        x1 = box_center[:, 0] - self.box_rad
        y1 = box_center[:, 1] - self.box_rad
        x2 = box_center[:, 0] + self.box_rad
        y2 = box_center[:, 1] + self.box_rad
        bbox = np.vstack((x1, y1, x2, y2)).transpose()
        for d in range(len(box_center)):
            crp = image.crop(bbox[d]).convert('RGB')
            crp = self.transform(crp)
            crop_obj.append(crp)
        crop_obj = torch.stack(crop_obj)
        return crop_obj

    def _build_graph(self, index):
        all_trip = np.zeros([0, 3], dtype=np.float32)
        for i in range(self.num_obj):
            for j in range(self.num_obj):
                trip = [i, 0, j]
                all_trip = np.vstack((all_trip, trip))
        return torch.FloatTensor(all_trip)


def dt_collate_fn(batch):
    """
    :return: src dst. each is a list with dict element
    - 'index': list of str with length N
    - 'image': list of FloatTensor in shape (Dt, V, 1, C, H, W)
    - 'crop': list of FloatTensor in shape (Dt, V, o, C, RH, RW)
    - 'bbox': (Dt, V, o, 4)
    - 'trip': (Dt, V, t, 3)
    """
    key_set = ['index', 'image', 'crop', 'bbox', 'trip', 'valid']
    all_batch = {}
    dt = len(batch[0])
    V = len(batch)
    for key in key_set:
        all_batch[key] = []

    for f in range(dt):
        for v in range(len(batch)):
            frame = batch[v][f]
            for key in key_set:
                all_batch[key].append(frame[key])

    for key in all_batch:
        if key == 'index':
            continue
        if key in ['image', 'crop']:
            tensor = torch.stack(all_batch[key])
            all_batch[key] = tensor.view(dt, V, -1, 3, tensor.size(-2), tensor.size(-1))
        elif key in ['bbox', 'trip', 'valid']:
            tensor = torch.stack(all_batch[key])
            all_batch[key] = tensor.view(dt, V, -1, tensor.size(-1))
        else:
            print('key not exist', key)
            raise KeyError

    return all_batch



