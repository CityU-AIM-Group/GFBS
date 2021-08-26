'''Train CIFAR10 with PyTorch.'''
import torch
import logging
import datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import sys
sys.path.append('./differentiable_models')

import torchvision.transforms as transforms

import os
import argparse
from differentiable_models import *
from utils import save_model, MODEL_DICT
import time
os.environ['CUDA_VISIBLE_DEVICE']='0'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=str(__file__)[:-3]+'_'+time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+'.log', level=logging.DEBUG, format=LOG_FORMAT, filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Data
def load_data(dataset):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == "cifar10":
        trainset = datasets.CIFAR10(root='./data', type='train+val', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        valset = datasets.CIFAR10(root='./data', type='test', transform=transform_val)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
    elif dataset == "cifar100":
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, valloader

# Training
def train(net, optimizer, dataloader, epoch, p):
    net.train()
    for i, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1)[1]
        acc = (pred == target).float().mean()

        if i % 100 == 0:
            logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.6f}, Accuracy: {:.4f}'.format(
                epoch, i, len(trainloader), loss.item(), acc.item()
            ))

# Testing
def test(net, dataloader, epoch, name):
    net.eval()
    test_loss = 0
    correct = 0
    global best_accuracy

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1)[1]
            correct += (pred == target).float().sum().item()

    test_loss /= len(dataloader.dataset)
    acc = correct / len(dataloader.dataset)
    logging.info('Val set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(
        test_loss, acc
    ))
    if acc > best_accuracy or epoch == args.epochs:
        # if epoch > 50: # Do not save the models before the 50th epoch
        logging.info("Saving the model.....")
        save_path = './checkpoints/'+name+'/epoch_{}_acc_{:.4f}.pth'.format(str(epoch), acc)
        save_model(net, acc, epoch, optimizer, scheduler, name, save_path)
                
        best_accuracy = acc

def train_and_evaluate(net, trainloader, testloader, optimizer, scheduler, total_epochs, start_epoch, name, p):
    # Without +1: 0~299; with +1: 1~300
    for epoch in range(start_epoch + 1, total_epochs + 1):

        # Run one epoch for both train and test
        logging.info("Epoch {}/{}".format(epoch, total_epochs))
        print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # compute number of batches in one epoch(one full pass over the training set)
        train(net, optimizer, trainloader, epoch, p)
        # logging.info('Learning_rate: %.4f' % (scheduler.get_last_lr()[0]))
        # writer.add_scalar('Learning_rate', epoch, torch.tensor(scheduler.get_last_lr()))
        
        scheduler.step()

        # Evaluate for one epoch on test set
        test(net, testloader, epoch, name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--net', default='vgg16', type=str, choices=list(MODEL_DICT.keys()), help='network used for training')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset used for training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--p', default=0.3, type=float, help='remaining channels')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--checkpoint', default=None, help='The checkpoint file (.pth)')
    parser.add_argument('--epochs', default=160, help='The number of training epochs')
    args = parser.parse_args()
    logging.info(args)

    trainloader, testloader = load_data(args.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # setup Tensorboard file path
    # writer = SummaryWriter('./summarys/'+args.net)

    net = MODEL_DICT[args.net].to(device)
    logging.info('==> Building model.. '+str(args.net)+str(net))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
    # The milestones mean update the lr AFTER the milestone epoch
    scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.2)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    if args.resume:
        logging.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        #
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_accuracy = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        #
    else:
        logging.info('==> Starting from scratch..')
        start_epoch = 0
    
    # Setup best accuracy for comparing and model checkpoints
    best_accuracy = 0.0

    # print summary of model
    # summary(net, (3, 32, 32))
    train_and_evaluate(net, trainloader, testloader, optimizer, scheduler, total_epochs=args.epochs, start_epoch=start_epoch, name=args.net, p=args.p)

    # writer.close()
