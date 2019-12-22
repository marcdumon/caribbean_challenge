# --------------------------------------------------------------------------------------------------------
# 2019/11/20
# src - data_sets.py
# md
# --------------------------------------------------------------------------------------------------------
import PIL
import torch as th
from PIL import Image
from torch.utils.data import Dataset
import cv2


class RoofDatasetPlus(Dataset):
    def __init__(self, items, meta_con, meta_cat, labels, transforms):
        fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/'
        self.fp_processed = f'{fp}processed/train_valid_test_augmented/'
        self.items = items
        self.meta_con = meta_con
        self.meta_cat = meta_cat
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        x = Image.open(self.fp_processed + item + '.png')
        x = self.transforms(x)
        x_con = self.meta_con[index]
        x_cat = self.meta_cat[index]
        y = self.labels[index]
        return x, x_con, x_cat, y


class RoofDataset(Dataset):

    def __init__(self, items: list, labels: list, transforms):
        fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/'
        self.fp_processed = f'{fp}processed/train_valid_test_augmented/'
        self.items = items
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        x = Image.open(self.fp_processed + item + '.png')
        # x = x.convert('RGB')  # RGBA to RGB
        x = self.transforms(x)
        y = self.labels[index]
        return x, y


if __name__ == '__main__':
    rd = RoofDataset(['7a1c4d74'], [1], None)
    print(rd.__getitem__(0))
