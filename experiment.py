# --------------------------------------------------------------------------------------------------------
# 2019/04/20
# 0_ml_project_template - experiment.py
# md
# --------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torchvision as thv
from pandas import CategoricalDtype, Categorical
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from models.roof_dataset import RoofDataset, RoofDatasetPlus
from models.callbacks import CallbackContainer, PrintLogs, TensorboardCB, SaveModel, SaveConfusionMatrix
from models.metrics import Accuracy, PredictionEntropy, MetricContainer, MetricCallback, Precision, Recall
from models.predict_model import predict_image
from models.trainer import Trainer
from models.model import SimpleCNN, ResnetPlus
from visualization.visualize import plot_confusion_matrix
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

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")

# --------------------------------------------------------------------------------------------------------
""" 
Here we run the experiment.
"""

# Parameters
params = {'image_size': 224, 'bs': 48,
          'n_epochs': 10, 'lr': 3e-4, 'momentum': 0.90, 'workers': 8}
params['experiment'] = f'Resnet152plus_Size{params["image_size"]}_Normal_25K_Metadata'
# params['experiment'] = f'test'
test_mode = True
log_on = False

# DATA
fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/'
train = pd.read_csv(fp + 'processed/train_plus.csv').sample(25000)
valid = pd.read_csv(fp + 'processed/valid_plus.csv')

if log_on:
    sys.stdout = open(f'../models/{params["experiment"]}.txt', 'w')

params['train_label_counts'] = train['label'].value_counts()
params['valid_label_counts'] = valid['label'].value_counts()
print('-' * 120, '\nTrain Labels count')
print(params['train_label_counts'])
print('-' * 120, '\nValid Labels count')
print(params['valid_label_counts'])

# Metadata
#   First handle nan, otherwise cat.code for nan is -1, resulting in error in ebedding (index out of range: -1)
train = train.fillna(0)
valid = valid.fillna(0)

countries = ['colombia', 'guatemala', 'st_lucia']
places = ['borde_rural', 'borde_soacha', 'castries', 'dennery', 'gros_islet', 'mixco_1_and_ebenezer', 'mixco_3']
labels = ['concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other']
params['labels'] = labels

countries_cat_type = CategoricalDtype(categories=countries, ordered=True)
places_cat_type = CategoricalDtype(categories=places, ordered=True)
labels_cat_type = CategoricalDtype(categories=labels + [0], ordered=True)  # +[0] for the nan's in neighbour labels

for df in [train, valid]:
    # Categorical
    df.loc[:, 'country'] = df.loc[:, 'country'].astype(str).astype(countries_cat_type).cat.codes
    df.loc[:, 'place'] = df.loc[:, 'place'].astype(places_cat_type).cat.codes
    df.loc[:, 'verified'] = df.loc[:, 'verified'].astype(int)
    for i in range(1, 20):
        df.loc[:, f'l_{i}'] = df.loc[:, f'l_{i}'].astype(labels_cat_type).cat.codes
    # Continuous
    df.loc[:, 'complexity'] = (df['complexity'] - df['complexity'].mean()) / df['complexity'].std()
    df.loc[:, 'area'] = (df['area'] - df['area'].mean()) / df['area'].std()
    mu = df.loc[:, 'd_1':'d_19'].values.mean()
    sigma = df.loc[:, 'd_1':'d_19'].values.std()
    for i in range(1, 20):
        df.loc[:, f'd_{i}'] = (df[f'd_{i}'] - mu) / sigma
    # Output
    df.loc[:, 'label'] = df.loc[:, 'label'].astype(labels_cat_type).cat.codes
    # valid.loc[:, 'label'] = valid.loc[:, 'label'].astype(labels_cat_type).cat.codes

train_items = train['id'].to_list()
train_meta_con = train[['complexity', 'area'] + [f'd_{i}' for i in range(1, 20)]].values
train_meta_cat = train[['country', 'place', 'verified'] + [f'l_{i}' for i in range(1, 20)]].values
train_labels = train['label'].to_list()

valid_items = valid['id'].to_list()
valid_meta_con = valid[['complexity', 'area'] + [f'd_{i}' for i in range(1, 20)]].values
valid_meta_cat = valid[['country', 'place', 'verified'] + [f'l_{i}' for i in range(1, 20)]].values
valid_labels = valid['label'].to_list()

transforms = transforms.Compose([
    transforms.Resize((params['image_size'], params['image_size'])),
    # transforms.CenterCrop(params['image_size']),
    transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = RoofDatasetPlus(train_items, train_meta_con, train_meta_cat, train_labels, transforms)
valid_ds = RoofDatasetPlus(valid_items, valid_meta_con, valid_meta_cat, valid_labels, transforms)


net = ResnetPlus()

# Unfreeze layers
# Todo: This works for resnet, but not for other models.
#  By default when we load a model, the parameters have requires_grad=True
# unfreeze_layers = [net.layer1, net.layer2, net.layer3, net.layer4]
# unfreeze_layers = [net.fc]
# for layer in unfreeze_layers:
#     for param in layer.parameters():
#         param.requires_grad = True
print('-' * 120)
for name, param in net.named_parameters():
    print(name, param.requires_grad)
print('-' * 120)
# print(summary(net, [(3, params['image_size'], params['image_size']), (22,), (21,)],
#               batch_size=params['bs'], device='cpu'))
# print('-' * 120)

# optimizer = Adam(net.parameters(), lr=params['lr'])
optimizer = Adam([{'params': net.resnet.layer4.parameters(), 'lr': params['lr'] / 3},
                  {'params': net.resnet.layer3.parameters(), 'lr': params['lr'] / 6},
                  {'params': net.resnet.layer2.parameters(), 'lr': params['lr'] / 9},
                  {'params': net.resnet.layer1.parameters(), 'lr': params['lr'] / 9}],
                 lr=params['lr'])
criterion = nn.CrossEntropyLoss()

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
if not test_mode:
    cbc.register(TensorboardCB(every_n_epoch=1, experiment_name=params['experiment']))
    cbc.register(SaveModel())
    cbc.register(SaveConfusionMatrix())
trainer = Trainer(train_ds=train_ds, valid_ds=valid_ds, model=net, criterion=criterion,
                  optimizer=optimizer, params=params, callbacks=cbc, metrics=mc)

model = trainer.train()
