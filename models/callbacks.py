# --------------------------------------------------------------------------------------------------------
# 2019/04/16
# 0_pytorch_tools - callbacks.py
# md
# --------------------------------------------------------------------------------------------------------
import datetime
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import torch.nn as nn
from visualization.visualize import plot_confusion_matrix


class CallbackContainer:

    def __init__(self, callbacks: list = None):
        self.callbacks = callbacks or []

    def register(self, callback):
        self.callbacks.append(callback)

    def on_train_begin(self, logs: dict = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs=logs)

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch=epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch=batch, logs=logs)

    def on_loss_begin(self):
        "Called after forward pass but before loss has been computed."
        for callback in self.callbacks:
            callback.on_loss_begin()

    def on_loss_end(self):
        for callback in self.callbacks:
            callback.on_loss_end()

    def on_backward_begin(self):
        "Called after the forward pass and the loss has been computed, but before backprop."
        for callback in self.callbacks:
            callback.on_backward_begin()

    def on_backward_end(self):
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        for callback in self.callbacks:
            callback.on_backward_end()

    def on_batch_end(self, batch: int, logs: dict = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch=batch, logs=logs)

    def on_epoch_end(self, epoch, logs: dict = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch=epoch, logs=logs)

    def on_train_end(self, logs: dict = None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs=logs)


class Callback:

    def __init__(self):
        self.params = None
        self.model = None

    def on_train_begin(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_loss_begin(self):
        pass

    def on_backward_begin(self):
        pass

    def on_backward_end(self):
        pass

    def on_loss_end(self):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_train_end(self, logs):
        pass


class SaveModel(Callback):
    """
    Saves the best and last model
    """

    def __init__(self):
        self.best_valid_loss = 1e9
        super(SaveModel, self).__init__()

    def on_train_begin(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        torch.save(logs['model'], f'../models/{logs["params"]["experiment"]}-last_model.pth.tar')
        valid_loss = sum(logs['valid_loss']) / len(logs['valid_loss'])
        self.best_valid_loss = min(self.best_valid_loss, valid_loss)
        if valid_loss == self.best_valid_loss:
            print(f'{now} Saving new best model')
            torch.save(logs['model'], f'../models/{logs["params"]["experiment"]}-best_model_{epoch}_{valid_loss}.pth.tar')

    def on_train_end(self, logs):
        pass

class PrintLogs(Callback):
    """
    Prints losses and metrics to console.
    """

    def __init__(self, every_n_epoch: int = 1):
        self.every_n_epoch = every_n_epoch
        self.train_begin_time = None
        self.train_loss = None
        self.n_train_batches = None
        self.n_valid_batches = None
        super(PrintLogs, self).__init__()

    def on_train_begin(self, logs):
        self.train_begin_time = datetime.datetime.now()
        print('{} Training starts'.format(self.train_begin_time.strftime('%Y-%m-%d %H:%M:%S')))

    def on_epoch_begin(self, epoch, logs):
        self.train_loss = 0

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        if epoch % self.every_n_epoch == 0:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            train_loss = sum(logs['train_loss']) / len(logs['train_loss'])
            valid_loss = sum(logs['valid_loss']) / len(logs['valid_loss'])
            # Get all metrics from logs. Metrics have the "metric." prefix
            metric_str = ''
            for k in logs:
                if 'metric.' in k: metric_str += '{}: {:6.4f} | '.format(k.split('.')[1], logs[k])
            print('{} {: 6d}/{} | loss (t/v): {:7.5f}/{:7.5f} | '
                  .format(now, epoch, logs['params']['n_epochs'], train_loss, valid_loss) + metric_str)

    def on_train_end(self, logs):
        now = datetime.datetime.now()
        total_time = now - self.train_begin_time
        tt = datetime.datetime(1, 1, 1) + total_time
        print('{} Training ends'.format(now.strftime('%Y-%m-%d %H:%M:%S')))
        print('Total training took: {} days, {} hours, {} minutes, {} seconds'.format(tt.day - 1, tt.hour, tt.minute, tt.second))

class TensorboardCB(Callback):
    """
    """
    # Todo: Tensorboard generates several numpy warnings: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated;
    def __init__(self, every_n_epoch: int = 1, tb_path='../tensorboard/', experiment_name=''):
        experiment_name = '{}/{}'.format(experiment_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=tb_path + experiment_name)
        self.every_n_epoch = every_n_epoch
        super(TensorboardCB, self).__init__()

    def on_train_begin(self, logs):
        # Add graph
        # Todo: Add_graph generate several jit warnings
        # Todo add_graph doesn't shows input "x_cat"
        # dummy_inpu r.flush()

        # Write params and model text
        params_str = '##Parameters:\n'
        for k, v in logs['params'].items():
            params_str += '- {}: {}\n'.format(k, v)
        params_str += '##Model:\n'
        for l in logs['model'].named_children():
            params_str += '- {}\n'.format(l)
        self.writer.add_text('Parameters', params_str)

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_loss_begin(self):
        "Called after forward pass but before loss has been computed."
        pass

    def on_loss_end(self):
        pass

    def on_backward_begin(self):
        "Called after the forward pass and the loss has been computed, but before backprop."
        pass

    def on_backward_end(self):
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass

    def on_batch_end(self, batch: int, logs: dict = None):
        pass

    def on_epoch_end(self, epoch, logs: dict = None):
        if epoch % self.every_n_epoch == 0:
            # Write losses and metrics
            train_loss = sum(logs['train_loss']) / len(logs['train_loss'])
            valid_loss = sum(logs['valid_loss']) / len(logs['valid_loss'])
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Valid', valid_loss, epoch)
            self.writer.add_scalar('Loss/Train-Valid', train_loss - valid_loss, epoch)

            # Get all metrics from logs. Metrics have the "metric." prefix
            for k, v in logs.items():
                if 'metric.' in k:
                    self.writer.add_scalar('Metrics/' + k, v, epoch)

            # Write the histograms for weights and biases and their gradients
            for name, param in logs['model'].named_parameters():
                name_parts = name.split('.')
                name = '{}/{}'.format(name_parts[0], name_parts[1])
                self.writer.add_histogram(name, param, epoch)
                if param.requires_grad:
                    self.writer.add_histogram(name + '_grad', param.grad, epoch)

    def on_train_end(self, logs: dict = None):
        pass

class SaveConfusionMatrix(Callback):
    """
    Saves a confusion matrix when valid loss reaches a new low
    """

    def __init__(self):
        self.best_valid_loss = 1e9
        super(SaveConfusionMatrix, self).__init__()

    def on_train_begin(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        valid_loss = sum(logs['valid_loss']) / len(logs['valid_loss'])
        self.best_valid_loss = min(self.best_valid_loss, valid_loss)
        if valid_loss == self.best_valid_loss:
            print(f'{now} Saving confusion matrix')
            plot_confusion_matrix(logs['y_true'], logs['y_pred'], labels=logs['params']['labels'],
                                  fname=f'../models/{logs["params"]["experiment"]}-best_model_{epoch}_{valid_loss}.png')

    def on_train_end(self, logs):
        pass
