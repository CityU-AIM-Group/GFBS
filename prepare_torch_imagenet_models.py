'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('./differentiable_models')
import os
import argparse
from differentiable_models import *
from utils import progress_bar, MODEL_DICT, save_model, save_model_net_only
import numpy as np
import time
import copy

try:
    from imagenet_dali import get_imagenet_iter_dali
except:
    pass

from misc import AverageMeter, accuracy, print_log
os.environ['CUDA_VISIBLE_DEVICE']='0'
# MODEL_DICT = {'dresnet20': DResNet20(), 'dresnet56': DResNet56(), 'vgg16': VGG('VGG16'), 'gatevgg16': GateVGG('GateVGG16')}
log = open(os.path.join('./{}.log'.format('resnet50_test')), 'w')
# Data
def load_data():
    print('==> Preparing data..')
    if args.dali:
        train_loader = get_imagenet_iter_dali(type='train', image_dir=args.data_dir, batch_size=256,
                                          num_threads=4, crop=224, device_id=0, num_gpus=1)
        val_loader = get_imagenet_iter_dali(type='val', image_dir=args.data_dir, batch_size=50,
                                          num_threads=4, crop=224, device_id=0, num_gpus=1)
    else:
        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
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
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=25, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    return train_loader, val_loader

def validate(model, criterion, val_loader, log):
    name = args.net
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        if args.dali:
            input = data[0]["data"].cuda(non_blocking=True)
            target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        else:
            input, target = data[0].cuda(), data[1].cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var).cuda()
        loss = criterion(output, target_var).cuda()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)
    if args.save:
        save_path = 'baseline_model_r50.pth'

        save_model_net_only(net, name, save_path)
    return top1.avg




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get the ImageNet pretrained model from torchvision for pruning')
    #################################################################################################################
    parser.add_argument('--net', default='resnet50', type=str, help='network used for training')
    parser.add_argument('--batch_size', '-b', default=512, type=int,  help='Batch Size')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--dali', action='store_true', help='use dali dataloader')
    parser.add_argument('--data_dir', default='/home/xliu423/imagenet', 
                    type=str, help='The main dir of ImageNet')
    #################################################################################################################
    parser.add_argument('--save', '-s', default=True, type=bool, help='save model or not')
    parser.add_argument('--checkpoint', default='./resnet50-19c8e357.pth', help='The checkpoint file (.pth)')
    args = parser.parse_args()
    global save
    save = args.save

    trainloader, testloader = load_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model
    net = MODEL_DICT[args.net]
    print('==> Building model.. ')
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    print('==> Testing the performance of a checkpoint..')
    if not os.path.isdir('checkpoint'):
        # download checkpoint from "https://download.pytorch.org/models/resnet50-19c8e357.pth"
        print('==> Downloading pretrained model..')
        import wget
        url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
        filename = wget.download(url)
    checkpoint = torch.load(args.checkpoint)
    net_dict = net.state_dict()
    pretrained_dict = checkpoint
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    best_accuracy = 0.0
    validate(net, criterion, testloader, log)
