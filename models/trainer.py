# --------------------------------------------------------------------------------------------------------
# 2019/04/18
# 0_ml_project_template - trainer.py
# md
# --------------------------------------------------------------------------------------------------------
import json
from time import sleep
from typing import Collection

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.backends import cudnn

from models.callbacks import Callback, CallbackContainer, TensorboardCB, PrintLogs
from models.metrics import MetricContainer, Accuracy, MetricCallback, PredictionEntropy

# device = torch.device('cuda:0')
device = th.device('cuda:0') if th.cuda.is_available() else th.device('cpu')
# device = th.device('cpu')
cudnn.benchmark = True  # should speed thing up ?


class Trainer:
    """
    Trainer class provides training loop.
    It needs training and validation dataset, model, criterion, optizer, callback container and metriccontainer.
    The params dict must have:
        - 'workers': 0,
        - 'bs': 64,
        - 'n_epochs': 1000,
        - 'lr': 1e-3,
        - 'momentum': 0.90
    """

    def __init__(self, train_ds: Dataset, valid_ds: Dataset, model: nn.Module, criterion, optimizer,
                 params: dict, callbacks: CallbackContainer, metrics: MetricContainer):
        self.model = model.to(device)
        self.params = params
        self.criterion = criterion
        self.optimizer = optimizer
        self.cbc = callbacks
        self.mc = metrics
        self.train_dl = self.make_data_loader(train_ds, bs=self.params['bs'])
        self.valid_dl = self.make_data_loader(valid_ds, bs=self.params['bs'])
        img, x_con, x_cat, _ = iter(self.train_dl).next()
        img, x_con, x_cat = img.to(device), x_con.float().to(device), x_cat.to(device)
        self.dummy_input = (img, x_con, x_cat)

    def train(self) -> nn.Module:  # Todo: Implement from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

        logs = {'params': self.params, 'model': self.model, 'dummy_input': self.dummy_input}

        self.cbc.on_train_begin(logs=logs)

        for epoch in range(1, self.params['n_epochs'] + 1):
            self.cbc.on_epoch_begin(epoch=epoch, logs=logs)
            logs['train_loss'] = []
            self.model.train()
            with th.set_grad_enabled(True):
                for batch, data in enumerate(self.train_dl, 1):
                    self.cbc.on_batch_begin(batch=batch, logs=logs)
                    x, x_con, x_cat, y_true = data
                    x, x_con, x_cat, y_true = x.to(device), x_con.float().to(device), x_cat.to(device), y_true.long().to(device)

                    self.optimizer.zero_grad()
                    y_pred = self.model(x, x_con, x_cat)
                    self.cbc.on_loss_begin()
                    loss = self.criterion(y_pred, y_true)
                    logs['train_loss'] = np.append(logs['train_loss'], loss.item())
                    self.cbc.on_loss_end()

                    self.cbc.on_backward_begin()
                    loss.backward()
                    self.optimizer.step()
                    self.cbc.on_backward_end()

                    self.cbc.on_batch_end(batch, logs=logs)

            # Validation
            self.model.eval()
            # with th.no_grad():
            with th.set_grad_enabled(False):
                logs['valid_loss'] = logs['y_true'] = np.array([])
                logs['y_pred'] = np.empty((0, 2))  # y_pred has the form of [a,b] with a+b=1 # todo !!!!!!!!!!!!!!!!!!!!!!!!!!!
                for valid_batch, data in enumerate(self.valid_dl, 1):
                    x, x_con, x_cat, y_true = data
                    x, x_con, x_cat, y_true = x.to(device), x_con.float().to(device), x_cat.to(device), y_true.long().to(device)
                    y_pred = self.model(x, x_con, x_cat)  # Not probabilities !!! Needs softmax to get probabilities
                    loss = self.criterion(y_pred, y_true)  # nn.CrossEntropyLoss() applies softmax on y_pred, so don't apply softmax on y_pred !
                    logs['valid_loss'] = np.append(logs['valid_loss'], [loss.item()])

                    # Store predictions for confusion matrix
                    y_pred = F.softmax(y_pred, dim=1)  # Only now change y_pred to probabilities
                    y_pred = th.argmax(y_pred, dim=1)
                    logs['y_true'] = np.append(logs['y_true'], y_true.cpu())
                    logs['y_pred'] = np.append(logs['y_pred'], y_pred.cpu())

            self.cbc.on_epoch_end(epoch=epoch, logs=logs)
        self.cbc.on_train_end(logs=logs)
        return self.model  # Trained model

    def make_data_loader(self, dataset: Dataset, bs) -> DataLoader:
        # Todo: move dataloaders to experiment.py
        """ put a Dataset into a Dataloader"""
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=self.params['workers'])
        # print('Set shuffle back on!!!!')
        # dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=self.params['workers'])
        return dataloader
