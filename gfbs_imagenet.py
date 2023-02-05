'''Train CIFAR10 with PyTorch.'''
import torch
import logging
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
import sys
sys.path.append('./differentiable_models')

import torchvision.transforms as transforms
from fvcore.nn import FlopCountAnalysis, flop_count_table

import os
import copy
import argparse
from differentiable_models import *
from utils import save_model, MODEL_DICT
import time
import datetime

try:
    from imagenet_dali import get_imagenet_iter_dali
except:
    pass

from misc import AverageMeter, accuracy
# os.environ['CUDA_VISIBLE_DEVICE']='0'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=str(__file__)[:-3]+'_'+time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+'.log', 
                    level=logging.INFO, 
                    format=LOG_FORMAT, 
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# MODEL_DICT = {'dresnet20': DResNet20(), 'dresnet56': DResNet56(), 'vgg16': VGG('VGG16'), 'maskedvgg16': MaskedVGG('MaskedVGG16')}
# TODO: Make apis for mobilenetv2
# Data
def load_data(data_dir, bs, workers, use_dali):
    if use_dali:
        train_loader = get_imagenet_iter_dali(type='train', image_dir=data_dir, batch_size=bs,
                                            num_threads=3, crop=224, device_id=0, num_gpus=1)
        val_loader = get_imagenet_iter_dali(type='val', image_dir=data_dir, batch_size=100,
                                            num_threads=3, crop=224, device_id=0, num_gpus=1)
    else:
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=100, shuffle=False,
            num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    total_epochs = 100
    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # nvmlInit()
        # deviceCount = nvmlDeviceGetCount()
        # for i in range(deviceCount):
        #     handle = nvmlDeviceGetHandleByIndex(i)
        #     print("GPU", i, ":", nvmlDeviceGetName(handle))
        # handle = nvmlDeviceGetHandleByIndex(0)
        # info = nvmlDeviceGetMemoryInfo(handle)
        # logging.info("Memory Total: {}".format(info.total))
        # logging.info("Memory Free: {}".format(info.free))
        # logging.info("Memory Used: {}".format(info.used))
        # logging.info("Temperature is %d C"%nvmlDeviceGetTemperature(handle,0))
        # logging.info("Fan speed is "nvmlDeviceGetFanSpeed(handle))
        # logging.info("Power ststus",nvmlDeviceGetPowerState(handle))
        # nvmlShutdown()
        # logging.info(psutil.virtual_memory())  # physical memory usage
        
        if args.dali:
            input = data[0]["data"].cuda(non_blocking=True)
            target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        else:
            input, target = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
        # measure data loading time

        data_time.update(time.time() - end)
        # train_loader_len = int(math.ceil(train_loader._size / args.train_bs))

        # target = target.cuda()
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        remain_time = batch_time.avg * len(train_loader) * (total_epochs - epoch) + \
                        batch_time.avg * (len(train_loader) - i)
        remain_time = str(datetime.timedelta(seconds=remain_time))
        
        if i % 20 == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                      'Remain {remain}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), 
                batch_time=batch_time, remain=remain_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

def test(val_loader, model, criterion, optimizer, scheduler, epoch, ratio):
    name = args.net
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    remaining, true_ratio = check_remaining_channels(model)
    pruned_ratio = 1. - true_ratio
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if args.dali:
                input = data[0]["data"].cuda(non_blocking=True)
                target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            else:
                input, target = data[0].cuda(non_blocking=True), data[1].cuda(non_blocking=True)
            

            # compute output
            output = model(input).cuda()
            loss = criterion(output, target).cuda()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                logging.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg))
    acc1 = top1.avg
    acc5 = top5.avg
    save_path = './checkpoints/'+name+'/baseline_acc_{:.3f}.pth'.format(top1.avg)

    return acc1

def flops(model, resolution=224):
    new_model = copy.deepcopy(model)
    device = next(new_model.parameters()).device
    tensor = (torch.rand(1,3,resolution,resolution, device=device), )
    # return 1
    flops = FlopCountAnalysis(new_model, tensor)
    del new_model
    return flops.total() / 1e6

def chn_mapper(acc_sort_idx, bn_dict, pruned_ratio):
    '''
    acc_sort_idx: a list of sorted index from large to small value
    bn_dict[Dict]: a dict with key as the name of the gate layer, the value as the total number of channels
    pruned_ratio: the ratio of the channels to be pruned
    '''
    mapper = list(bn_dict.keys())
    bgn = 0
    modules = []
    # modules: a list of list, each list contains the start and end index of the channels
    for bn_layer in bn_dict:
        end = bgn + bn_dict[bn_layer] - 1
        channel_idx = [bgn, end]
        bgn = end + 1
        modules.append(channel_idx)

    # find the index of the channels to be removed
    rm_chns = int(len(acc_sort_idx) * (1 - pruned_ratio))
    Stop = False
    while not Stop:
        last_to_remove = acc_sort_idx[rm_chns:][0]
        # find the corresponding gate layer
        for idx, layer in enumerate(modules):
            if layer[0] <= last_to_remove <= layer[1]:
                if 'gate3' in mapper[idx]:
                    rm_chns += 1
                else:
                    Stop = True
            
    toremove = acc_sort_idx[rm_chns:] # large to small importance
    dic = {}
    dic_count = {}
    for channel in toremove:
        for idx, layer in enumerate(modules):
            if layer[0] <= channel <= layer[1]: # find the corresponding gate layer
                if mapper[idx] not in dic:
                    dic[mapper[idx]] = []
                    dic[mapper[idx]].append(channel - layer[0])
                    dic_count[mapper[idx]] = 1
                else:
                    if dic_count[mapper[idx]] < bn_dict[mapper[idx]] - 1: # Avoid Layer Collapse
                        dic[mapper[idx]].append(channel - layer[0])
                        dic_count[mapper[idx]] += 1
    return dic, dic_count

def bn2gatevgg(name):
    l = name.split('.')
    if len(l) == 4: # module.features.1.weight ==> module.features.2.gate
        l[-1] = 'gate'
        l[-2] = str(int(l[-2]) + 1)
    elif len(l) == 3: # module.features.1 ==> module.features.2.gate
        l[-1] = str(int(l[-1]) + 1)
        l.append('gate')
    return '.'.join(l)

def bn2gateresnet(name):
    l = name.split('.')
    if len(l)>1:
        if 'bn' in l[-2]: # module.layer2.4.bn2.weight => module.layer2.4.gate2.gate
            l[-1] = 'gate'
            l[-2] = str('gate' + l[-2][-1])
        elif 'bn' in l[-1]: # module.layer1.0.bn1 ==> module.layer1.0.gate1.gate
            l[-1] = str('gate' + l[-1][-1]) # bn1 ==> gate1
            l.append('gate')
    else:
        l[-1] = str('gate' + l[-1][-1]) # bn1 ==> gate1
        l.append('gate')
    return '.'.join(l)

def bn2mobilenet(name):
    l = name.split('.')
    if 'bn' in l[-2]: # module.layers.2.bn2.weight => module.layer.2.gate2.gate
        l[-1] = 'gate'
        l[-2] = str('gate' + l[-2][-1])
    elif 'bn' in l[-1]: # module.layer.1.bn1 ==> module.layer.1.gate1.gate
        l[-1] = str('gate' + l[-1][-1]) # bn1 ==> gate1
        l.append('gate')
    return '.'.join(l)

class Gate(torch.nn.Module):
    def __init__(self, out_planes):
        super(Gate, self).__init__()
        self.gate = nn.Parameter(torch.ones(1, out_planes, 1, 1), requires_grad=False)

    def forward(self, x):
        return self.gate * x

down_inchn_dict = {}
def map_gate_to_convbn_resnet(net, gate_layer_name, remove_dict, device_id):
    '''
    net: The network
    gate_layer_name: The name of the gate layer
    remove_index_list: The index of the channels to be removed
    '''
    # Unwrap the model
    net = net.module if hasattr(net, 'module') else net
    gate_layer_name = gate_layer_name.split('module.')[1] if 'module.' in gate_layer_name else gate_layer_name
    remove_dict = {k.split('module.')[1]: v for k, v in remove_dict.items()} if 'module.' in list(remove_dict.keys())[0] else remove_dict

    remove_index_list = remove_dict[gate_layer_name]
    l = gate_layer_name.split('.') 
    
    old_gate = net._modules # Loop through the model dict{dict{dict{dict{ .. }}}
    for key in l[:-1]:
        key = int(key) if key.isdigit() else key
        try:
            old_gate = old_gate[key]
        except:
            old_gate = old_gate._modules[key]
        
    original_list_length = old_gate.gate.size(1)

    # Get the indices of remaining indexes 
    preserve_index_list = [i for i in range(original_list_length) if i not in remove_index_list]

    # replace gate
    #############################
    new_gate = Gate(int(len(preserve_index_list))).cuda()
    #############################
    new_gate.gate.data = old_gate.gate.data[:, preserve_index_list]
    assert sum(new_gate.gate.data[0, :, 0, 0]) == original_list_length - len(remove_index_list)
    assert sum(new_gate.gate.data[0, :, 0, 0]) == sum(old_gate.gate.data[0, preserve_index_list, 0, 0])
    print(gate_layer_name)
    if 'layer' not in gate_layer_name:
        net._modules[l[0]] = new_gate
        print("replace {}, length {} ==> {}".format(l[0], original_list_length, len(preserve_index_list)))
    else:
        net._modules[l[0]]._modules[l[1]]._modules[l[2]] = new_gate
        print("replace {}, length {} ==> {}".format('.'.join(l[:-1]), original_list_length, len(preserve_index_list)))

    assert original_list_length > len(remove_index_list)
    l = l[:-1] # layer2.4.gate2.gate ==> layer2.4.bn2
    l[-1] = str('bn' + l[-1][-1]) # bn
    bn_layer_name = '.'.join(l)
    l[-1] = str('conv' + l[-1][-1]) # conv
    conv_layer_name = '.'.join(l)

    # replace previous conv
    old_conv = net._modules # Loop through the model dict{dict{dict{dict{ .. }}}
    for key in conv_layer_name.split('.'):
        key = int(key) if key.isdigit() else key
        try:
            old_conv = old_conv[key]
        except:
            old_conv = old_conv._modules[key]

    #############################
    new_conv = nn.Conv2d(old_conv.in_channels, int(len(preserve_index_list)), kernel_size=old_conv.kernel_size, \
        stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None).cuda()
    #############################
    new_conv.weight.data = old_conv.weight.data[preserve_index_list]
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data[preserve_index_list]

    conv_layer_name = conv_layer_name.split('.')
    if len(conv_layer_name) == 1: 
        net._modules[conv_layer_name[0]] = new_conv
        print("replace {} ==> {}".format(conv_layer_name[0], new_conv))
    else:
        net._modules[conv_layer_name[0]]._modules[conv_layer_name[1]]._modules[conv_layer_name[2]] = new_conv
        print("replace {} ==> {}".format('.'.join([conv_layer_name[0], conv_layer_name[1], conv_layer_name[2]]), new_conv))

    # replace previous bn
    old_bn = net._modules # Loop through the model dict{dict{dict{dict{ .. }}}
    for key in bn_layer_name.split('.'):
        key = int(key) if key.isdigit() else key
        try:
            old_bn = old_bn[key]
        except:
            old_bn = old_bn._modules[key]
        
    #############################
    new_bn = nn.BatchNorm2d(int(len(preserve_index_list))).cuda()
    #############################
    new_bn.weight.data = old_bn.weight.data[preserve_index_list]
    new_bn.bias.data = old_bn.bias.data[preserve_index_list]
    new_bn.running_mean = old_bn.running_mean[preserve_index_list]
    new_bn.running_var = old_bn.running_var[preserve_index_list]

    bn_layer_name = bn_layer_name.split('.')
    if len(bn_layer_name) == 1: 
        net._modules[bn_layer_name[0]] = new_bn
        print("replace {} ==> {}".format(bn_layer_name[0], new_bn))
    else:
        net._modules[bn_layer_name[0]]._modules[bn_layer_name[1]]._modules[bn_layer_name[2]] = new_bn
        print("replace {} ==> {}".format('.'.join([bn_layer_name[0], bn_layer_name[1], bn_layer_name[2]]), new_bn))

    gate_name_list_ordered = list(remove_dict.keys())
    gate_index = gate_name_list_ordered.index(gate_layer_name)
    next_index = gate_index + 1
    if next_index < len(gate_name_list_ordered):
        next_conv_name = gate_name_list_ordered[next_index]
        next_conv_name = next_conv_name.split('.')[:-1]
        next_conv_name[-1] = next_conv_name[-1].replace('gate', 'conv')
        next_conv_name = '.'.join(next_conv_name)
    else:
        next_conv_name = 'module.fc'
                
    if '0.conv1' in next_conv_name:
        global down_inchn_dict
        ds_name = next_conv_name.replace('conv1', 'downsample.0')
        nc_name = next_conv_name.split('.')
        down_inchn_dict[ds_name] = int(net._modules[nc_name[0]]._modules[nc_name[1]]._modules[nc_name[2]].weight.data.size(1))

    if next_conv_name != 'module.fc':
        old_conv = net._modules # Loop through the model dict{dict{dict{dict{ .. }}}
        for key in next_conv_name.split('.'):
            key = int(key) if key.isdigit() else key
            try:
                old_conv = old_conv[key]
            except:
                old_conv = old_conv._modules[key]

        #############################
        new_conv = nn.Conv2d(int(len(preserve_index_list)), \
                old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding).cuda()
        #############################
        new_conv.weight.data = old_conv.weight.data[:, preserve_index_list, :, :] # The second dimension is the input, [out, in, k, k]
        next_conv_name = next_conv_name.split('.')
        net._modules[next_conv_name[0]]._modules[next_conv_name[1]]._modules[next_conv_name[2]] = new_conv
        print("replace {} ==> {}".format('.'.join([next_conv_name[0], next_conv_name[1], next_conv_name[2]]), new_conv))
    else:
        old_conv = net._modules['fc']

        #############################
        new_conv = nn.Linear(int(len(preserve_index_list)), old_conv.out_features).cuda()
        #############################
        new_conv.weight.data = old_conv.weight.data[:, preserve_index_list]
        net._modules['fc'] = new_conv
        print("replace {} ==> {}".format('fc', new_conv))

    ## Also apply channel pruning on downsample layers
    if '.0.gate3' in gate_layer_name: # downsample block at the beginning of each layer
        gate_layer_name = '.'.join(gate_layer_name.split('.')[:-1])
        conv1_name = gate_layer_name.replace('gate3', 'conv1')
        conv1 = net._modules # Loop through the model dict{dict{dict{dict{ .. }}}
        for key in conv1_name.split('.'):
            key = int(key) if key.isdigit() else key
            try:
                conv1 = conv1[key]
            except:
                conv1 = conv1._modules[key]
        in_channels = int(conv1.in_channels)
        out_channels = int(len(preserve_index_list))

        downsample_conv_name = gate_layer_name.replace('gate3', 'downsample.0')
        old_conv = net._modules # Loop through the model dict{dict{dict{dict{ .. }}}
        for key in downsample_conv_name.split('.'):
            key = int(key) if key.isdigit() else key
            try:
                old_conv = old_conv[key]
            except:
                old_conv = old_conv._modules[key]

        #############################
        new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=old_conv.kernel_size, \
            stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None).cuda()
        #############################
        gate1_name = gate_layer_name.replace('gate3', 'gate1.gate')
        gate1_index = gate_name_list_ordered.index(gate1_name)
        gate1_prev_gate_index = gate1_index - 1
        gate1in_remove_list = remove_dict[gate_name_list_ordered[gate1_prev_gate_index]]
        # Index based on the full length of the input channels
        print(down_inchn_dict)
        gate1in_preserve_index_list = [i for i in range(down_inchn_dict[downsample_conv_name]) if i not in gate1in_remove_list]
        assert len(gate1in_preserve_index_list) == in_channels, "len(gate1in_preserve_index_list) = {}, in_channels = {}".format(len(gate1in_preserve_index_list), in_channels)
        assert len(preserve_index_list) == out_channels, "len(preserve_index_list) = {}, out_channels = {}".format(len(preserve_index_list), out_channels)
        new_conv.weight.data = old_conv.weight.data[preserve_index_list, :, :, :] # out channels
        new_conv.weight.data = new_conv.weight.data[:, gate1in_preserve_index_list, :, :] # in channels
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data[preserve_index_list]

        net._modules[downsample_conv_name.split('.')[0]]._modules[downsample_conv_name.split('.')[1]]._modules[downsample_conv_name.split('.')[2]]._modules[downsample_conv_name.split('.')[3]] = new_conv
        print("replace {} ==> {}".format('.'.join([downsample_conv_name.split('.')[0], downsample_conv_name.split('.')[1], downsample_conv_name.split('.')[2], downsample_conv_name.split('.')[3]]), new_conv))

        downsample_bn_name = gate_layer_name.replace('gate3', 'downsample.1').replace('module.', '')
        old_bn = net._modules # Loop through the model dict{dict{dict{dict{ .. }}}
        for key in downsample_bn_name.split('.'):
            key = int(key) if key.isdigit() else key
            try:
                old_bn = old_bn[key]
            except:
                old_bn = old_bn._modules[key]

        #############################
        new_bn = nn.BatchNorm2d(out_channels).cuda()
        #############################
        new_bn.weight.data = old_bn.weight.data[preserve_index_list]
        new_bn.bias.data = old_bn.bias.data[preserve_index_list]
        new_bn.running_mean.data = old_bn.running_mean.data[preserve_index_list]
        new_bn.running_var.data = old_bn.running_var.data[preserve_index_list]

        net._modules[downsample_bn_name.split('.')[0]]._modules[downsample_bn_name.split('.')[1]]._modules[downsample_bn_name.split('.')[2]]._modules[downsample_bn_name.split('.')[3]] = new_bn
        print("replace {} ==> {}".format('.'.join([downsample_bn_name.split('.')[0], downsample_bn_name.split('.')[1], downsample_bn_name.split('.')[2], downsample_bn_name.split('.')[3]]), new_bn))
        # The downsample should have input channels: conv1.in_channels and output channels: conv3.out_channels
    # print(net)
    print('X' * 100)
    # Wrap the model with nn.DataParallel again
    net = torch.nn.DataParallel(net, device_ids=device_id)
    return net


def check_remaining_channels(net):
    total_remain = 0
    total = 0
    for name, param in net.named_parameters():
        if 'gate' in name:
            # print("Name: {}, Channels: {}".format(name, torch.sum(param).item()))
            total_remain += torch.sum(param)
            total += param.shape[1]
    # print('channels remain: %d' % total_remain.item())
    percentage = 100 * (total_remain.item()/total)
    print('Remaining percentage: %.2f (%d/%d)' % (percentage, total_remain.item(), total))
    return total_remain, total_remain.item()/total



def finetune_and_evaluate(net, criterion, trainloader, testloader, optimizer, scheduler, total_epochs, start_epoch, name, ratio, smooth):
    # Without +1: 0~299; with +1: 1~300
    best_accuracy = 0.0
    scheduler.step(0)
    for epoch in range(start_epoch, total_epochs):

        # Run one epoch for both train and test
        logging.info("Epoch {}/{}".format(epoch, total_epochs))
        print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # compute number of batches in one epoch(one full pass over the training set)
        # finetune_train(net, optimizer, trainloader, epoch)
        train(trainloader, net, criterion, optimizer, epoch)
        # logging.info('Learning_rate: %.4f' % (scheduler.get_last_lr()[0]))
        # writer.add_scalar('Learning_rate', epoch, torch.tensor(scheduler.get_last_lr()))
        
        scheduler.step(epoch)

        # Evaluate for one epoch on test set
        acc = test(testloader, net, criterion, optimizer, scheduler, epoch, ratio)
        # if total_epochs in [45, 160]: # save model at the final finetune stage
        if acc >= best_accuracy and acc >= 0.6:
            logging.info("Saving the model.....")
            if not os.path.isdir('checkpoints/'+name):
                os.mkdir('checkpoints/'+name)
            save_path = './checkpoints/'+name+'gfbs_acc_{:.4f}_chnratio_{:.2f}.pth'.format(acc, ratio)
            save_model(net, acc, epoch, optimizer, scheduler, name, save_path)
            best_accuracy = acc
                
    return best_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Importance of Each Layer')
    parser.add_argument('--p', default=0.4, type=float, help='channel pruned ratio')
    parser.add_argument('--smooth', '-s', action='store_true', help='finetune the network for several epochs after the pruning of each layer')
    parser.add_argument('--data_dir', default='/home/xliu423/imagenet', help='The path of dataset')
    parser.add_argument('--beta', default=True, help='use beta information or not')
    parser.add_argument('--w_beta', default=0.05, type=float, help='beta weight')
    parser.add_argument('--checkpoint', default='./baseline_model_r50.pth', help='The checkpoint file (.pth)')
    parser.add_argument('--epochs', default=120, help='The number of training epochs')
    parser.add_argument('--train_bs', default=512, type=int, help='The number of training batch size')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    ############################################################################################
    # scheduler
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=1e-2, type=float, help='weight decay')
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler')
    parser.add_argument('--warmup-epochs', default=5, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('--cooldown-epochs', default=10, type=int, metavar='N', help='number of cooldown epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    ############################################################################################
    parser.add_argument('--dali', action='store_true', help='use dali')
    parser.add_argument('--net', default='resnet50', type=str, help='network used for training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument(
                    '--gpu',
                    type=str,
                    default='0,1',
                    help='Select gpu to use')
                    
    args = parser.parse_args()
    logging.info(args)

    trainloader, testloader = load_data(args.data_dir, args.train_bs, args.workers, args.dali)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda', "Only support GPU training"

    net = MODEL_DICT[args.net]
    flops_base = flops(net, 224)
    params_base = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info("Baseline FLOPS: {} M".format(str(flops(net, 224))))
    logging.info("Baseline Params: {} M".format(str(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6)))
    logging.info('==> Building model.. '+str(args.net)+str(net))

    torch.manual_seed(args.seed)
    if device == 'cuda':
        device_id = []
        for i in args.gpu.split(','):
            device_id.append(int(i))
        logging.info("==> Using GPU {}".format(','.join(list(map(str, device_id)))))
        # net = torch.nn.DataParallel(net, device_ids=device_id).cuda()
        net = net.cuda()
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    
    # if args.checkpoint:
    if args.checkpoint:
        logging.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint)['net']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        info = net.load_state_dict(checkpoint)
        logging.info(info)

    optimizer1 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)
    net.train()
    
    data = next(iter(trainloader))
    if args.dali:
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    else:
        images, labels = data[0].cuda(), data[1].cuda()
    
    optimizer1.zero_grad()
    output = net(images)
    loss = F.cross_entropy(output, labels)
    loss.backward()

    # use a simpler way to add capability for higher version of torch
    gamma_grad_dict = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            grad_weight = m.weight.grad.abs().clone().detach().data 
            gamma_grad_dict[name] = grad_weight / torch.norm(grad_weight, 2)

    # map the saliency values from name of BN to the name of following gate
    assert 'resnet' in args.net
    gamma_grad_dict = dict((bn2gateresnet(k), v) for (k, v) in gamma_grad_dict.items())

    # check if the model is data parallel
    if 'module' in list(net.state_dict().keys())[0]:
        del gamma_grad_dict['module.layer1.0.downsample.1']
        del gamma_grad_dict['module.layer2.0.downsample.1']
        del gamma_grad_dict['module.layer3.0.downsample.1']
        del gamma_grad_dict['module.layer4.0.downsample.1']
    else:
        del gamma_grad_dict['layer1.0.downsample.1']
        del gamma_grad_dict['layer2.0.downsample.1']
        del gamma_grad_dict['layer3.0.downsample.1']
        del gamma_grad_dict['layer4.0.downsample.1']
        
    ############################## Get toremove channels ##############################
    gate_dict = {}
    gamma_dict = {}
    gamma_list = []
    beta_dict = {}
    beta_list = {}
    for named_params in net.named_parameters():
        name, params = named_params
        # if isinstance(m, nn.BatchNorm2d): # select bn weights
        if 'downsample' not in name:
            if 'weight' in name and len(params.shape) == 1:
                if 'vgg' in args.net:
                    name = bn2gatevgg(name)
                elif 'resnet' in args.net:
                    name = bn2gateresnet(name)
                elif 'mobilenetv2' in args.net:
                    name = bn2mobilenet(name)

                gate_dict[name] = int(params.shape[0])
                gammas_pre_norm = params.abs().clone().detach()
                
                gammas_norm = gammas_pre_norm / torch.norm(gammas_pre_norm, 2) # Norm gamma
                gamma_dict[name] = gammas_norm
            if 'bias' in name and len(params.shape) == 1 and 'classifier' not in name:
                if 'vgg' in args.net:
                    name = bn2gatevgg(name)
                elif 'resnet' in args.net:
                    name = bn2gateresnet(name)
                elif 'mobilenetv2' in args.net:
                    name = bn2mobilenet(name)
                if name in gamma_dict.keys(): # Remove Conv2d biases
                    betas_pre_norm = params.clone().detach()
                    betas_norm = betas_pre_norm / torch.norm(betas_pre_norm, 2)
                    beta_dict[name] = betas_norm

    logging.info('Total number of channels for each gate: ' + str(gate_dict))
    assert gamma_dict.keys() == beta_dict.keys() == gamma_grad_dict.keys()
    assert len(gate_dict.keys()) == len(gamma_dict.keys())

    dicts = [gamma_dict, gamma_grad_dict, beta_dict] # ensure the last conv of each residual block to have averaged saliency scores
    for d in dicts:
        avg = (d['layer1.0.gate3.gate'] + d['layer1.1.gate3.gate'] + d['layer1.2.gate3.gate']) / 3
        d['layer1.0.gate3.gate'] = avg
        d['layer1.1.gate3.gate'] = avg
        d['layer1.2.gate3.gate'] = avg
        avg = (d['layer2.0.gate3.gate'] + d['layer2.1.gate3.gate'] + d['layer2.2.gate3.gate'] + d['layer2.3.gate3.gate']) / 4
        d['layer2.0.gate3.gate'] = avg
        d['layer2.1.gate3.gate'] = avg
        d['layer2.2.gate3.gate'] = avg
        d['layer2.3.gate3.gate'] = avg
        avg = (d['layer3.0.gate3.gate'] + d['layer3.1.gate3.gate'] + d['layer3.2.gate3.gate'] + d['layer3.3.gate3.gate'] + d['layer3.4.gate3.gate'] + d['layer3.5.gate3.gate']) / 6
        d['layer3.0.gate3.gate'] = avg
        d['layer3.1.gate3.gate'] = avg
        d['layer3.2.gate3.gate'] = avg
        d['layer3.3.gate3.gate'] = avg
        d['layer3.4.gate3.gate'] = avg
        d['layer3.5.gate3.gate'] = avg
        avg = (d['layer4.0.gate3.gate'] + d['layer4.1.gate3.gate'] + d['layer4.2.gate3.gate']) / 3
        d['layer4.0.gate3.gate'] = avg
        d['layer4.1.gate3.gate'] = avg
        d['layer4.2.gate3.gate'] = avg

    # *************** Get GFBS for BN ****************
    # gamma_dict: a dict that contains the gamma values for each layer
    # gamma_grad_dict: a dict that contains the grad of the gamma values for each layer
    for gate_layer in gamma_dict.keys():
        assert gate_layer in gamma_grad_dict
        assert gate_layer in beta_dict
        assert gamma_dict[gate_layer].shape == gamma_grad_dict[gate_layer].shape == beta_dict[gate_layer].shape
        taylor = gamma_dict[gate_layer].cpu() * gamma_grad_dict[gate_layer].cpu()
        ############################### Whether to employ beta information
        if args.beta:
            taylor += beta_dict[gate_layer].cpu() * args.w_beta
        gamma_list.extend(taylor) # total length is the sum of all channels
    # **************************************************

    # sort the gamma_list and get the index from largest value to the smallest value
    acc_sort_idx = sorted(range(len(gamma_list)), key=lambda k: gamma_list[k])[::-1]
    remove_dic, remove_dic_count = chn_mapper(acc_sort_idx, gate_dict, args.p)
    del gamma_dict
    del beta_dict
    del gamma_grad_dict
    ######################### If print to remove channels, uncommit this line #############
    # logging.info(remove_dic)
    #######################################################################################
    remove_dic_count_new = {}
    for name, m in net.named_parameters():
        if name in remove_dic_count:
            remove_dic_count_new[name] = remove_dic_count[name]
    logging.info('Total remove channel amount for each gate: ' + str(remove_dic_count_new))
    del remove_dic_count

    ############################## Remove channels ########################################
    # sort remove_dic according to the occurrence in the model
    remove_dic_new = {}
    for name, m in net.named_parameters():
        if name in remove_dic:
            remove_dic_new[name] = remove_dic[name]
    logging.info('Total remove channel indexes for each gate: ' + str(remove_dic_new))
    del remove_dic
            
    del loss
    del output
    del images
    del labels
    del data
    del optimizer1
    torch.cuda.empty_cache()
    
    info = net.load_state_dict(checkpoint)
    logging.info(info)

    # Start Pruning
    logging.info("X" * 50)
    for gate_layer in remove_dic_new.keys():
        assert remove_dic_count_new[gate_layer] < gate_dict[gate_layer]
        # for channel in remove_dic_new[gate_layer]:
        #     net.state_dict()[gate_layer][:, channel, :, :].data.copy_(torch.zeros_like(net.state_dict()[gate_layer][:, channel, :, :].data))
        logging.info('Finished removing {} channels in '.format(remove_dic_count_new[gate_layer])+str(gate_layer)+', remaining {}, applying to the network ... ' \
            .format(gate_dict[gate_layer]-remove_dic_count_new[gate_layer]))

        net = map_gate_to_convbn_resnet(net, gate_layer, remove_dic_new, device_id)
        # logging.info(net)

        # profile the forward pass
        # print the results

        if args.smooth:
            if 'gate3' in gate_layer:
                criterion = nn.CrossEntropyLoss().cuda()
                logging.info('Finished removing channels in '+str(gate_layer)+', finetune for several epochs.')
                optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
                scheduler = MultiStepLR(optimizer2, milestones=[5, 8], gamma=0.1)
                start_epoch = 0
                best_accuracy = finetune_and_evaluate(net, criterion, trainloader, testloader, optimizer2, scheduler, total_epochs=10, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
                logging.info('Best accuracy: {:.4f}'.format(best_accuracy))

    logging.info('Finished removing')
    logging.info(net)
    
    net(torch.randn(10, 3, 224, 224).cuda(non_blocking=True))
    
    flops_after_prune = flops(net, 224)
    logging.info("Pruned FLOPS: {} M".format(str(flops(net, 224))))
    logging.info("Pruned Params: {} M".format(str(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6)))
    logging.info("FLOPS pruned ratio: {:.4f}".format(1. - flops_after_prune / flops_base))
    logging.info("Params pruned ratio: {:.4f}".format(1. - sum(p.numel() for p in net.parameters() if p.requires_grad) / params_base))

    if args.smooth:
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer2, milestones=[10, 15], gamma=0.2)
        start_epoch = 0
        best_accuracy = finetune_and_evaluate(net, criterion, trainloader, testloader, optimizer2, scheduler, total_epochs=45, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
        logging.info('Best accuracy: {:.4f}'.format(best_accuracy))
    else:
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                        momentum=0.9, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer2, milestones=[30, 60, 90], gamma=0.1)
        start_epoch = 0
        best_accuracy = finetune_and_evaluate(net, criterion, trainloader, testloader, optimizer2, scheduler, total_epochs=args.epochs, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
        logging.info('Best accuracy: {:.4f}'.format(best_accuracy))
