'''Train CIFAR10 with PyTorch.'''
import torch
import logging
import datasets
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import sys
sys.path.append('./differentiable_models')

import torchvision.transforms as transforms

import os
import copy
import argparse
from differentiable_models import *
from utils import save_model, MODEL_DICT, CosineAnnealingLR
import time
os.environ['CUDA_VISIBLE_DEVICE']='0'
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
def load_data(dataset, bs):
    print('==> Preparing data..{}'.format(dataset))
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.CIFAR10(root='./data', type='train+val', transform=transform_train, download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)

        valset = datasets.CIFAR10(root='./data', type='val', transform=transform_test, download=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=200, shuffle=False, num_workers=2)

        testset = datasets.CIFAR10(root='./data', type='test', transform=transform_test, download=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    elif dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=bs, shuffle=True, num_workers=2)

        valset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=bs, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, valloader, testloader

# Training
def gfbs(net, optimizer, dtloader, epoch):
    net.train()
    # global bn_grad_list
    for i, (data, target) in enumerate(dtloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # optimizer.step()
        pred = output.max(1)[1]
        acc = (pred == target).float().mean()

        if i % 100 == 0:
            logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.6f}, Accuracy: {:.4f}'.format(
                epoch, i, len(dtloader), loss.item(), acc.item()
            ))
        
        break
    return net

def test(net, dataloader):
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
    # logging.info('Val set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(
    #     test_loss, acc
    # ))
    if acc > best_accuracy:
        best_accuracy = acc
    return test_loss, acc

def mapper(acc_sort_idx, bn_dict, pruned_ratio):
    toremove = acc_sort_idx[int(len(acc_sort_idx) * (1 - pruned_ratio)):]
    bgn = 0
    modules = []
    for bn_layer in bn_dict:
        end = bgn + bn_dict[bn_layer] - 1
        channel_idx = [bgn, end]
        bgn = end + 1
        modules.append(channel_idx)
    mapper = list(bn_dict.keys())
    dic = {}
    dic_count = {}
    for channel in toremove:
        for idx, layer in enumerate(modules):
            if layer[0] <= channel <= layer[1]:
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
    if 'bn' in l[-2]: # module.layer2.4.bn2.weight => module.layer2.4.gate2.gate
        l[-1] = 'gate'
        l[-2] = str('gate' + l[-2][-1])
    elif 'bn' in l[-1]: # module.layer1.0.bn1 ==> module.layer1.0.gate1.gate
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


def finetune_train(net, optimizer, dataloader, epoch):
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
def finetune_test(net, dataloader, optimizer, scheduler, epoch, name, ratio, smooth):
    net.eval()
    test_loss = 0
    correct = 0
    global best_accuracy
    # global acc_list_for_betas
    remaining, true_ratio = check_remaining_channels(net)
    pruned_ratio = 1. - true_ratio
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
    if acc > best_accuracy:
        if ratio - pruned_ratio < 0.05: # Do not save the models before the 50th epoch
            logging.info('Pruned ratio / Desire pruned ratio: {:.2f}/{:.2f}'.format(pruned_ratio, ratio))
            logging.info("Saving the model.....")
            if smooth:
                if not os.path.isdir('checkpoints/'+name+'/smooth/'):
                    os.mkdir('checkpoints/'+name+'/smooth/')
                save_path = './checkpoints/'+name+'/smooth/'+'gfbsv2_smooth_epoch_{}_acc_{:.4f}_rem_{:.2f}_des_{:.2f}.pth'.format(str(epoch), acc, true_ratio, ratio)
            else:
                save_path = './checkpoints/'+name+'/gfbsv2_epoch_{}_acc_{:.4f}_r_{:.2f}.pth'.format(str(epoch), acc, true_ratio)
            save_model(net, acc, epoch, optimizer, scheduler, name, save_path)
                
            best_accuracy = acc

def finetune_and_evaluate(net, trainloader, testloader, optimizer, scheduler, total_epochs, start_epoch, name, ratio, smooth):
    # Without +1: 0~299; with +1: 1~300
    for epoch in range(start_epoch + 1, total_epochs + 1):

        # Run one epoch for both train and test
        logging.info("Epoch {}/{}".format(epoch, total_epochs))
        print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # compute number of batches in one epoch(one full pass over the training set)
        finetune_train(net, optimizer, trainloader, epoch)
        # logging.info('Learning_rate: %.4f' % (scheduler.get_last_lr()[0]))
        # writer.add_scalar('Learning_rate', epoch, torch.tensor(scheduler.get_last_lr()))
        
        scheduler.step()

        # Evaluate for one epoch on test set
        finetune_test(net, testloader, optimizer, scheduler, epoch, name, ratio, smooth)

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, grad_in, grad_out):
        grad_weight = grad_in[1]
        grad_weight = grad_weight.abs().clone().detach()
        grad_weight_norm = grad_weight / torch.norm(grad_weight, 2)
        grad_bias = grad_in[2]
        grad_bias = grad_bias.clone().detach()
        grad_bias_norm = grad_bias / torch.norm(grad_bias, 2)
        self.grad_weight = grad_weight_norm
        self.grad_bias = grad_bias_norm
        self.grad_out = grad_out
    def close(self):
        self.hook.remove()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Importance of Each Layer')
    parser.add_argument('--net', default='gatevgg16', type=str, choices=list(MODEL_DICT.keys()), help='network used for training')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset used for training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--p', default=0.7, type=float, help='channel pruned ratio')
    parser.add_argument('--smooth', '-s', action='store_true', help='finetune the network for 10 epochs after the pruning of each layer')
    parser.add_argument('--beta', default=True, help='use beta information or not')
    parser.add_argument('--beta_only', action='store_true', help='use beta information or not')
    parser.add_argument('--cosine', action='store_true', help='use cosine lr rate')
    parser.add_argument('--w_beta', default=0.05, type=float, help='beta weight')
    parser.add_argument('--checkpoint', default=None, help='The checkpoint file (.pth)')
    parser.add_argument('--epochs', default=300, help='The number of training epochs')
    parser.add_argument('--bs', default=128, type=int, help='The number of training epochs')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    logging.info(args)

    trainloader, valloader, testloader = load_data(args.dataset, args.bs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = MODEL_DICT[args.net].to(device)
    logging.info('==> Building model.. '+str(args.net)+str(net))

    # Setup best accuracy for comparing and model checkpoints
    best_accuracy = 0.90
    torch.manual_seed(args.seed)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    
    if args.checkpoint:
        logging.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])

    optimizer1 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)

    # global bn_grad_list
    # bn_grad_list = []
    # handle_list = []
    hook_dic = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # print('instance weight', m.weight)
            hook_dic[name] = Hook(m, backward=True)
            print('Register hook for ', name, ' with size ', m.weight.size())
            # m.weight.register_hook(grad_hook)
            # handle = m.weight.register_hook(grad_hook)
            # handle_list.append(handle)

    net = gfbs(net, optimizer1, valloader, 1)
    if 'vgg' in args.net:
        gamma_grad_dict = dict((bn2gatevgg(k), v.grad_weight) for (k, v) in hook_dic.items())
    elif 'resnet' in args.net:
        gamma_grad_dict = dict((bn2gateresnet(k), v.grad_weight) for (k, v) in hook_dic.items())
    elif 'mobilenet' in args.net:
        gamma_grad_dict = dict((bn2mobilenet(k), v.grad_weight) for (k, v) in hook_dic.items())
    else:
        raise NotImplementedError
    # for hook in hook_dic.keys():
        # print(hook)
        # print(hook_dic[hook].grad_weight)
    # for handle in handle_list: # Release memory by removing handles
    #     handle.remove()
    # for named_params in net.named_parameters():
    #     name, params = named_params
    #     print(name)
    ############################## Get toremove channels ##############################
    gate_dict = {}
    gamma_dict = {}
    gamma_list = []
    beta_dict = {}
    beta_list = {}
    for named_params in net.named_parameters():
        name, params = named_params
        if 'weight' in name and len(params.shape) == 1:
            if 'vgg' in args.net:
                name = bn2gatevgg(name)
            elif 'resnet' in args.net:
                name = bn2gateresnet(name)

            gate_dict[name] = int(params.shape[0])
            gammas_pre_norm = params.abs().clone().detach()
            
            gammas_norm = gammas_pre_norm / torch.norm(gammas_pre_norm, 2) # Norm gamma
            gamma_dict[name] = gammas_norm
        if 'bias' in name and len(params.shape) == 1 and 'classifier' not in name:
            if 'vgg' in args.net:
                name = bn2gatevgg(name)
            elif 'resnet' in args.net:
                name = bn2gateresnet(name)
            if name in gamma_dict.keys(): # Remove Conv2d biases
                betas_pre_norm = params.clone().detach()
                betas_norm = betas_pre_norm / torch.norm(betas_pre_norm, 2)
                beta_dict[name] = betas_norm

    logging.info(gate_dict)

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
        if args.beta_only:
            taylor = beta_dict[gate_layer].cpu() * args.w_beta
        gamma_list.extend(taylor)
    # **************************************************
    
    # Release hooks
    for k, hook in hook_dic.items():
        hook.close()

    # bn_grad_list = bn_grad_list[::-1]
    # for lyrs, chns in enumerate(bn_grad_list):
    #     assert chns.shape[0] == gate_dict[list(gate_dict)[lyrs]] # Check if the hooks match the gate params
    # bn_grad_list = [layer_bn_grad.abs() / torch.norm(layer_bn_grad.abs(), 2) for layer_bn_grad in bn_grad_list] # Norm grad
    # bn_grad_list_cat = torch.cat(bn_grad_list).tolist()
    # gamma_list = [a * b for a, b in zip(gamma_list, bn_grad_list_cat)]
    # bn_grad_list.clear()

    acc_sort_idx = sorted(range(len(gamma_list)), key=lambda k: gamma_list[k])[::-1]
    # logging.info(acc_sort_idx)
    remove_dic, remove_dic_count = mapper(acc_sort_idx, gate_dict, args.p)
    ######################### If print to remove channels, uncommit this line #############
    # logging.info(remove_dic)
    #######################################################################################
    logging.info('Total remove channel amount: ' + str(dict(sorted(remove_dic_count.items()))))

    ############################## Remove channels ##############################
    remove_dic = dict(sorted(remove_dic.items()))
    for gate_layer in remove_dic:
        assert remove_dic_count[gate_layer] < gate_dict[gate_layer]
        for channel in remove_dic[gate_layer]:
            net.state_dict()[gate_layer][:, channel, :, :].data.copy_(torch.zeros_like(net.state_dict()[gate_layer][:, channel, :, :].data))
        if args.smooth:
            logging.info('Finished removing channels in '+str(gate_layer)+', finetune for several epochs.')
            if args.dataset == 'cifar10':
                best_accuracy = 0.90
                optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
                scheduler = MultiStepLR(optimizer2, milestones=[5, 10], gamma=0.1)
            elif args.dataset == 'cifar100':
                best_accuracy = 0.68
                optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
                scheduler = MultiStepLR(optimizer2, milestones=[5, 10], gamma=0.2)
                if args.cosine:
                    scheduler = CosineAnnealingLR(optimizer2, 5, 30, 5, 0.0)
            start_epoch = 0
            finetune_and_evaluate(net, trainloader, testloader, optimizer2, scheduler, total_epochs=30, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
            logging.info('Best accuracy: {:.4f}'.format(best_accuracy))

    logging.info('Finished removing')
    optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer2, milestones=[10, 15], gamma=0.2)
    if args.cosine:
        scheduler = CosineAnnealingLR(optimizer2, 5, 30, 5, 0.0)
    start_epoch = 0
    finetune_and_evaluate(net, trainloader, testloader, optimizer2, scheduler, total_epochs=45, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
    logging.info('Best accuracy: {:.4f}'.format(best_accuracy))
    # state = {
    #         'net': net.state_dict(),
    #     }
    # if not os.path.isdir('checkpoints/'):
    #     os.mkdir('checkpoints/')
    # if not os.path.isdir('checkpoints/'+args.net):
    #     os.mkdir('checkpoints/'+args.net)
    # save_path = 'checkpoints/'+args.net+'/gfbspruned.pth'
    
    if not args.smooth:
        if args.dataset == 'cifar100':
            best_accuracy = 0.68
            acc_list_for_betas = []
            optimizer3 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                            momentum=0.9, weight_decay=5e-4)
            scheduler = MultiStepLR(optimizer3, milestones=[90, 180, 240], gamma=0.2)
            start_epoch = 0
            finetune_and_evaluate(net, trainloader, testloader, optimizer3, scheduler, total_epochs=args.epochs, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
            logging.info('Best accuracy: {:.4f}'.format(best_accuracy))
