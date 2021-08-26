'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

import torch
def save_model(net, acc, epoch, optimizer, scheduler, name, save_path):
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
    if not os.path.isdir('checkpoints/'):
        os.mkdir('checkpoints/')
    if not os.path.isdir('checkpoints/'+name):
        os.mkdir('checkpoints/'+name)
    torch.save(state, save_path)

def save_model_net_only(net, name, save_path):
    state = {
            'net': net.state_dict()
        }
    if not os.path.isdir('checkpoints/'):
        os.mkdir('checkpoints/')
    if not os.path.isdir('checkpoints/'+name):
        os.mkdir('checkpoints/'+name)
    torch.save(state, save_path)

import math, torch
import torch.nn as nn
from bisect import bisect_right
from torch.optim import Optimizer
class _LRScheduler(object):

  def __init__(self, optimizer, warmup_epochs, epochs):
    if not isinstance(optimizer, Optimizer):
      raise TypeError('{:} is not an Optimizer'.format(type(optimizer).__name__))
    self.optimizer = optimizer
    for group in optimizer.param_groups:
      group.setdefault('initial_lr', group['lr'])
    self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
    self.max_epochs = epochs
    self.warmup_epochs  = warmup_epochs
    self.current_epoch  = 0
    self.current_iter   = 0

  def extra_repr(self):
    return ''

  def __repr__(self):
    return ('{name}(warmup={warmup_epochs}, max-epoch={max_epochs}, current::epoch={current_epoch}, iter={current_iter:.2f}'.format(name=self.__class__.__name__, **self.__dict__)
              + ', {:})'.format(self.extra_repr()))

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)

  def get_lr(self):
    raise NotImplementedError

  def get_min_info(self):
    lrs = self.get_lr()
    return '#LR=[{:.6f}~{:.6f}] epoch={:03d}, iter={:4.2f}#'.format(min(lrs), max(lrs), self.current_epoch, self.current_iter)

  def get_min_lr(self):
    return min( self.get_lr() )

  def update(self, cur_epoch, cur_iter):
    if cur_epoch is not None:
      assert isinstance(cur_epoch, int) and cur_epoch>=0, 'invalid cur-epoch : {:}'.format(cur_epoch)
      self.current_epoch = cur_epoch
    if cur_iter is not None:
      assert isinstance(cur_iter, float) and cur_iter>=0, 'invalid cur-iter : {:}'.format(cur_iter)
      self.current_iter  = cur_iter
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
      param_group['lr'] = lr

class CosineAnnealingLR(_LRScheduler):

  def __init__(self, optimizer, warmup_epochs, epochs, T_max, eta_min):
    self.T_max = T_max
    self.eta_min = eta_min
    super(CosineAnnealingLR, self).__init__(optimizer, warmup_epochs, epochs)

  def extra_repr(self):
    return 'type={:}, T-max={:}, eta-min={:}'.format('cosine', self.T_max, self.eta_min)

  def get_lr(self):
    lrs = []
    for base_lr in self.base_lrs:
      if self.current_epoch >= self.warmup_epochs and self.current_epoch < self.max_epochs:
        last_epoch = self.current_epoch - self.warmup_epochs
        #if last_epoch < self.T_max:
        #if last_epoch < self.max_epochs:
        lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * last_epoch / self.T_max)) / 2
        #else:
        #  lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.T_max-1.0) / self.T_max)) / 2
      elif self.current_epoch >= self.max_epochs:
        lr = self.eta_min
      else:
        lr = (self.current_epoch / self.warmup_epochs + self.current_iter / self.warmup_epochs) * base_lr
      lrs.append( lr )
    return lrs

from differentiable_models import *
MODEL_DICT = {'gateresnet20': GateResNet20(), 
              'resnet50': resnet50(),
              'gateresnet32': GateResNet32(), 
              'gateresnet56': GateResNet56(), 
              'gateresnet110': GateResNet110(), 
              'gatevgg13': GateVGG('GateVGG13'), 
              'gatevgg16': GateVGG('GateVGG16'), 
              'gatemobilenetv2': GateMobileNetV2(),
              'gatedensenet40': GateDenseNet40(),
              'gateresnet32_c100': GateResNet32_c100(),
              'gateresnet56_c100': GateResNet56_c100()
              }