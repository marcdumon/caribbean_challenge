# --------------------------------------------------------------------------------------------------------
# 2019/04/20
# 0_ml_project_template - experiment.py
# md
# --------------------------------------------------------------------------------------------------------
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torchvision as thv
from pandas import CategoricalDtype, Categorical
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from models.roof_dataset import ImageDataset, MetaDataset
from models.callbacks import CallbackContainer, PrintLogs, TensorboardCB, SaveModel, SaveConfusionMatrix
from models.metrics import Accuracy, PredictionEntropy, MetricContainer, MetricCallback, Precision, Recall
from models.predict_model import predict_image
from models.trainer import Trainer
from models.nn_models import SimpleCNN, ResnetPlus, MetaData, Resnet
from visualization.visualize import plot_confusion_matrix, print_model_details
import pretrainedmodels
import time
from torchsummary import summary

pd.options.display.max_columns = 20
pd.options.display.width = 0
pd.set_option('expand_frame_repr', True)
pd.options.mode.chained_assignment = None

# To surpress anoying pytorch warning:
# UserWarning: Couldn't retrieve source code for container of type XXX. It won't be checked for correctness upon loading.
#   "type " + obj.__name__ + ". It won't be checked "
# See: https://discuss.pytorch.org/t/got-warning-couldnt-retrieve-source-code-for-container/7689/7
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
# Reproducability
seed = 42
np.random.seed(seed)
th.manual_seed(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

now = datetime.now()

# --------------------------------------------------------------------------------------------------------
# Settings
experiment_info = ''
test_mode_on = True
log_on = False
model_name = 'resnet1'

# Parameters
image_size = 256
bs = 40
lr = 1e-5
n_epochs = 10
n_samples = 0  # 0 -> all samples


# --------------------------------------------------------------------------------------------------------
def run_resnet1(t, v):  # All layers unfrozen, 1 LR
    tfms = transforms.Compose([transforms.Resize((image_size, image_size)),
                               # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    train_items, train_labels = t['id_aug'].values, t['label'].values
    valid_items, valid_labels = v['id_aug'].values, v['label'].values

    train_ds = ImageDataset(train_items, train_labels, tfms)
    valid_ds = ImageDataset(valid_items, valid_labels, tfms)
    net = Resnet(dropout=0)
    print_model_details(net, params)
    optimizer = Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(train_ds=train_ds, valid_ds=valid_ds, model=net, criterion=criterion,
                      optimizer=optimizer, params=params, callbacks=cbc, metrics=mc)
    trainer.train()


def run_resnet2(t, v):  # All layers frozen, except fc
    tfms = transforms.Compose([transforms.Resize((image_size, image_size)),
                               # transforms.CenterCrop(params['image_size']),
                               transforms.ToTensor(),
                               # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    train_items, train_labels = t['id_aug'].values, t['label'].values
    valid_items, valid_labels = v['id_aug'].values, v['label'].values

    train_ds = ImageDataset(train_items, train_labels, tfms)
    valid_ds = ImageDataset(valid_items, valid_labels, tfms)
    net = Resnet(dropout=0.)
    for layer in [net.resnet.layer1, net.resnet.layer2, net.resnet.layer3, net.resnet.layer4]:
        for param in layer.parameters():
            param.requires_grad = False
    print_model_details(net, params)
    optimizer = Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(train_ds=train_ds, valid_ds=valid_ds, model=net, criterion=criterion,
                      optimizer=optimizer, params=params, callbacks=cbc, metrics=mc)
    trainer.train()


def run_resnet3(t, v):  # All layers frozen, except layer4 and fc
    tfms = transforms.Compose([transforms.Resize((image_size, image_size)),
                               # transforms.CenterCrop(params['image_size']),
                               transforms.ToTensor(),
                               # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    train_items, train_labels = t['id_aug'].values, t['label'].values
    valid_items, valid_labels = v['id_aug'].values, v['label'].values

    train_ds = ImageDataset(train_items, train_labels, tfms)
    valid_ds = ImageDataset(valid_items, valid_labels, tfms)
    net = Resnet()
    for layer in [net.resnet.layer1, net.resnet.layer2, net.resnet.layer3]:
        for param in layer.parameters():
            param.requires_grad = False
    print_model_details(net, params)
    optimizer = Adam([{'params': net.resnet.layer4.parameters(), 'lr': 1e-5}],
                     lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(train_ds=train_ds, valid_ds=valid_ds, model=net, criterion=criterion,
                      optimizer=optimizer, params=params, callbacks=cbc, metrics=mc)
    trainer.train()


experiment_name = f'{model_name}_{image_size}_{n_samples}_{bs}_{lr}'
experiment_name += f'_{now.year}{now.month}{now.day}_{now.hour}{now.minute}{now.second}'
if log_on: sys.stdout = open(f'../models/logs/{experiment_name}.txt', 'w')
print(f'Starting: {experiment_name}')

params = {'experiment': experiment_name, 'image_size': image_size, 'bs': bs, 'n_epochs': n_epochs,
          'labels': ['concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other']}

# DATA
fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/'
train = pd.read_csv(fp + 'processed/train.csv')
valid = pd.read_csv(fp + 'processed/valid.csv')
if n_samples: train = train.sample(min(n_samples, len(train)))

print('-' * 120, '\nTrain Labels count')
print(train['label'].value_counts())
print('-' * 120, '\nValid Labels count')
print(valid['label'].value_counts())

# Metrics and Callbacks
# Todo: Repair precisision/recall
acc = Accuracy()
entr = PredictionEntropy()
precision = Precision()
recall = Recall()

mc = MetricContainer(metrics=[acc, precision, recall])
cbc = CallbackContainer()
# cbc.register(MetricCallback(mc))
cbc.register(PrintLogs(every_n_epoch=1))
if not test_mode_on:
    cbc.register(TensorboardCB(every_n_epoch=1, experiment_name=params['experiment']))
    cbc.register(SaveModel())
    cbc.register(SaveConfusionMatrix())

if model_name == 'resnet1':
    run_resnet1(train, valid)

if model_name == 'resnet2':
    run_resnet2(train, valid)
