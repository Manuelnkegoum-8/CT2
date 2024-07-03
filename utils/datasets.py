import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from skimage.io import imread
# from skimage import transform, color
from skimage import color
import pickle
import torch
# import cv2
import collections
import json


class COCODataset(Dataset):
    def __init__(
        self,
        image_size=224,
        split="train",
        dataset_dir='',
        mask_num=4,
        mask_random=False,
        n_cls=313,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.root_dir = os.path.join(self.dataset_dir, split)       # for imagenet
        self.image_size = image_size
        self.split = split
        self.mask_num = mask_num
        self.mask_random = mask_random
        self.n_cls = n_cls
        self.imgs = os.listdir(self.root_dir)
        self.mask_L = np.zeros((mask_num, 313)).astype(np.bool_) 
        assert os.path.exists(os.path.join('./', 'mask_prior.pickle'))
        fp = open(os.path.join('./', 'mask_prior.pickle'), 'rb')
        L_dict = pickle.load(fp)
        self.mask_L = np.zeros((mask_num, 313)).astype(np.bool_)     # [4, 313]
        for key in range(101):
            for ii in range(mask_num):
                start_key = ii * (100//mask_num)      # 0
                end_key = (ii+1)* (100//mask_num)     # 25
                if start_key <= key < end_key:
                    self.mask_L[ii, :] += L_dict[key].astype(np.bool_)
                    break
            
        self.mask_L = self.mask_L.astype(np.float32)
        #self.random_mask_L = np.random.randn(self.mask_L.shape)

    def rgb_to_lab(self, img):
        assert img.dtype == np.uint8
        return color.rgb2lab(img).astype(np.float32)

    def numpy_to_torch(self, img):
        tensor = torch.from_numpy(np.moveaxis(img, -1, 0))      # [c, h, w]
        return tensor.type(torch.float32)

    def get_img(self, idx):
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        w, h = img.size
        if self.split == 'train':
            img_transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            img_transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
            ])
        img_resized = img_transform(img)
        img_resized = np.array(img_resized)

        l_resized = self.rgb_to_lab(img_resized)[:, :, :1]
        ab_resized = self.rgb_to_lab(img_resized)[:, :, 1:]     # np.float32
        
        original_l = l_resized[:, :, 0]
        l = original_l.reshape((self.image_size * self.image_size))
        mask_p_c = np.zeros((self.image_size**2, self.n_cls), dtype=np.float32)
        for l_range in range(self.mask_num):
            start_l1, end_l1 = l_range * (100//self.mask_num), (l_range + 1) * (100 // self.mask_num)
            if end_l1 == 100:
                index_l1 = np.where((l >= start_l1) & (l <= end_l1))[0]
            else:
                index_l1 = np.where((l >= start_l1) & (l < end_l1))[0]

            if not self.mask_random:
                mask_p_c[index_l1, :] = self.mask_L[l_range, :]
            else:
                mask_p_c[index_l1, :] = self.random_mask_L[l_range, :]
        
        mask = torch.from_numpy(mask_p_c)
        img_l = self.numpy_to_torch(l_resized)
        img_ab = self.numpy_to_torch(ab_resized)
        return img_l, img_ab, mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_l, img_ab, mask = self.get_img(idx)
        return {"L": img_l, "ab": img_ab, "mask": mask}
