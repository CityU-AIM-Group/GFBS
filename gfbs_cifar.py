'''Train CIFAR10 with PyTorch.'''
import torch
import logging
import datasets
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import sys
sys.path.append('./differentiable_models')

import torchvision.transforms as transforms
from fvcore.nn import FlopCountAnalysis

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

        valset = datasets.CIFAR10(root='./data', type='val', transform=transform_train, download=True)
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
    if acc > best_accuracy:
        best_accuracy = acc
    return test_loss, acc

def flops(model, resolution=32):
    new_model = copy.deepcopy(model)
    device = next(new_model.parameters()).device
    tensor = (torch.rand(1,3,resolution,resolution, device=device), )
    # return 1
    flops = FlopCountAnalysis(new_model, tensor)
    del new_model
    return flops.total() / 1e6

def mapper(acc_sort_idx, bn_dict, pruned_ratio):
    '''
    acc_sort_idx: a list of sorted index from large to small value
    bn_dict[Dict]: a dict with key as the name of the gate layer, the value as the total number of channels
    pruned_ratio: the ratio of the channels to be pruned
    '''
    toremove = acc_sort_idx[int(len(acc_sort_idx) * (1 - pruned_ratio)):] # the index of the channels to be removed
    bgn = 0
    modules = []
    
    # modules: a list of list, each list contains the start and end index of the channels
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

class Gate(torch.nn.Module):
    def __init__(self, out_planes):
        super(Gate, self).__init__()
        self.gate = nn.Parameter(torch.ones(1, out_planes, 1, 1), requires_grad=False)

    def forward(self, x):
        return self.gate * x

def map_gate_to_convbn_vgg(net, gate_layer_name, remove_index_list, device):
    '''
    net: The network
    gate_layer_name: The name of the gate layer
    remove_index_list: The index of the channels to be removed
    '''
    # conv_layer_name is changing module.features.2.gate to module.features.0
    l = gate_layer_name.split('.') # module.features.2.gate
    original_list_length = net.module._modules[l[-3]][int(l[-2])].gate.size(1)
    preserve_index_list = [i for i in range(original_list_length) if i not in remove_index_list]

    # replace gate
    old_gate = net.module._modules[l[-3]][int(l[-2])]
    new_gate = Gate(int(len(preserve_index_list))).to(device)
    new_gate.gate.data = old_gate.gate.data[:, preserve_index_list]
    assert sum(new_gate.gate.data[0, :, 0, 0]) == original_list_length - len(remove_index_list)
    assert sum(new_gate.gate.data[0, :, 0, 0]) == sum(old_gate.gate.data[0, preserve_index_list, 0, 0])
    net.module._modules[l[-3]][int(l[-2])] = new_gate

    print("original_list_length: ", original_list_length)
    assert original_list_length > len(remove_index_list)
    l = l[1:][:-1] # features.2
    l[-1] = str(int(l[-1]) - 1) # bn
    bn_layer_name = '.'.join(l)
    l[-1] = str(int(l[-1]) - 1) # conv
    conv_layer_name = '.'.join(l)
    print("gate_layer_name: ", gate_layer_name)
    print("bn_layer_name: ", bn_layer_name)
    print("conv_layer_name: ", conv_layer_name)

    # replace previous conv
    old_conv = net.module._modules[conv_layer_name.split('.')[-2]][int(conv_layer_name.split('.')[-1])]
    new_conv = nn.Conv2d(old_conv.in_channels, int(len(preserve_index_list)), kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding).to(device)
    new_conv.weight.data = old_conv.weight.data[preserve_index_list]
    new_conv.bias.data = old_conv.bias.data[preserve_index_list]
    net.module._modules[conv_layer_name.split('.')[-2]][int(conv_layer_name.split('.')[-1])] = new_conv

    # replace previous bn
    old_bn = net.module._modules[bn_layer_name.split('.')[-2]][int(bn_layer_name.split('.')[-1])]
    new_bn = nn.BatchNorm2d(int(len(preserve_index_list))).to(device)
    new_bn.weight.data = old_bn.weight.data[preserve_index_list]
    new_bn.bias.data = old_bn.bias.data[preserve_index_list]
    new_bn.running_mean = old_bn.running_mean[preserve_index_list]
    new_bn.running_var = old_bn.running_var[preserve_index_list]
    net.module._modules[bn_layer_name.split('.')[-2]][int(bn_layer_name.split('.')[-1])] = new_bn

    # replace following conv
    Flag = False
    for name, module in net.named_modules():
        if name == '.'.join(gate_layer_name.split('.')[:-1]):
            Flag = True
        if Flag:
            if isinstance(module, nn.Conv2d):                
                next_conv_name = name
                break
            else:
                next_conv_name = 'module.classifier'
                
    if next_conv_name != 'module.classifier':
        old_conv = net.module._modules[next_conv_name.split('.')[-2]][int(next_conv_name.split('.')[-1])]
        new_conv = nn.Conv2d(int(len(preserve_index_list)), \
                old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding).to(device)
        new_conv.weight.data = old_conv.weight.data[:, preserve_index_list, :, :]
        net.module._modules[next_conv_name.split('.')[-2]][int(next_conv_name.split('.')[-1])] = new_conv
        print("next_conv_name: ", next_conv_name)
    else:
        old_conv = net.module._modules['classifier']
        new_conv = nn.Linear(int(len(preserve_index_list)), old_conv.out_features).to(device)
        new_conv.weight.data = old_conv.weight.data[:, preserve_index_list]
        net.module._modules['classifier'] = new_conv
        print("next_conv_name: ", next_conv_name)

    return net


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
def finetune_test(net, dataloader, optimizer, scheduler, epoch, name, ratio, smooth, best_accuracy):
    net.eval()
    test_loss = 0
    correct = 0
    
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
    return best_accuracy, acc

def finetune_and_evaluate(net, trainloader, testloader, optimizer, scheduler, total_epochs, start_epoch, name, ratio, smooth):
    # Without +1: 0~299; with +1: 1~300
    best_accuracy = 0.0
    for epoch in range(start_epoch + 1, total_epochs + 1):

        # Run one epoch for both train and test
        logging.info("Epoch {}/{}".format(epoch, total_epochs))
        print("Current time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        finetune_train(net, optimizer, trainloader, epoch)
        scheduler.step()

        # Evaluate for one epoch on test set
        best_accuracy, acc = finetune_test(net, testloader, optimizer, scheduler, epoch, name, ratio, smooth, best_accuracy)
        if total_epochs in [45, 160]: # save model at the final finetune stage
            if acc > best_accuracy and acc >= 0.9:
                logging.info("Saving the model.....")
                if not os.path.isdir('checkpoints/'+name+'/smooth/'):
                    os.mkdir('checkpoints/'+name+'/smooth/')
                save_path = './checkpoints/'+name+'/smooth/'+'gfbs_acc_{:.4f}_chnratio_{:.2f}.pth'.format(acc, ratio)
                save_model(net, acc, epoch, optimizer, scheduler, name, save_path)
                    
                best_accuracy = acc
    return best_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the Importance of Each Layer')
    parser.add_argument('--net', default='gatevgg16', type=str, choices=list(MODEL_DICT.keys()), help='network used for training')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset used for training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--p', default=0.7, type=float, help='channel pruned ratio')
    parser.add_argument('--smooth', '-s', action='store_true', help='finetune the network for several epochs after the pruning of each layer')
    parser.add_argument('--beta', default=True, help='use beta information or not')
    parser.add_argument('--beta_only', action='store_true', help='use beta information or not')
    parser.add_argument('--cosine', action='store_true', help='use cosine lr rate')
    parser.add_argument('--w_beta', default=0.05, type=float, help='beta weight')
    parser.add_argument('--checkpoint', default='./checkpoints', help='The checkpoint file (.pth)')
    parser.add_argument('--epochs', default=160, help='The number of training epochs')
    parser.add_argument('--bs', default=128, type=int, help='The number of training epochs')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    logging.info(args)

    trainloader, valloader, testloader = load_data(args.dataset, args.bs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = MODEL_DICT[args.net].to(device)
    flops_base = flops(net, 32 if args.dataset == 'cifar10' else 224)
    logging.info("FLOPS: {} M".format(str(flops(net, 32 if args.dataset == 'cifar10' else 224))))
    logging.info("Params: {} M".format(str(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6)))
    logging.info('==> Building model.. '+str(args.net)+str(net))

    # Setup best accuracy for comparing and model checkpoints
    best_accuracy = 0.90
    torch.manual_seed(args.seed)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    
    # if args.checkpoint:
    logging.info('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.checkpoint + '/' + args.net + '/model_best.pth')
    info = net.load_state_dict(checkpoint['net'])
    logging.info(info)

    optimizer1 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)

    net.train()
    for i, (data, target) in enumerate(valloader):
        data, target = data.to(device), target.to(device)

        optimizer1.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        break

    # use a simpler way to add capability for higher version of torch
    gamma_grad_dict = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            grad_weight = m.weight.grad.abs().clone().detach().data 
            gamma_grad_dict[name] = grad_weight / torch.norm(grad_weight, 2)

    # map the saliency values from name of BN to the name of following gate
    if 'vgg' in args.net:
        gamma_grad_dict = dict((bn2gatevgg(k), v) for (k, v) in gamma_grad_dict.items())
    elif 'resnet' in args.net:
        gamma_grad_dict = dict((bn2gateresnet(k), v) for (k, v) in gamma_grad_dict.items())
    elif 'mobilenet' in args.net:
        gamma_grad_dict = dict((bn2mobilenet(k), v) for (k, v) in gamma_grad_dict.items())
    else:
        raise NotImplementedError
        
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

    logging.info('Total number of channels for each gate: ' + str(gate_dict))

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
        gamma_list.extend(taylor) # total length is the sum of all channels
    # **************************************************

    # sort the gamma_list and get the index from largest value to the smallest value
    acc_sort_idx = sorted(range(len(gamma_list)), key=lambda k: gamma_list[k])[::-1]
    remove_dic, remove_dic_count = mapper(acc_sort_idx, gate_dict, args.p)
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

    # Start Pruning
    logging.info("X" * 50)
    for gate_layer in remove_dic_new.keys():
        assert remove_dic_count_new[gate_layer] < gate_dict[gate_layer]
        for channel in remove_dic_new[gate_layer]:
            net.state_dict()[gate_layer][:, channel, :, :].data.copy_(torch.zeros_like(net.state_dict()[gate_layer][:, channel, :, :].data))
        logging.info('Finished removing {} channels in '.format(remove_dic_count_new[gate_layer])+str(gate_layer)+', remaining {}, applying to the network ... ' \
            .format(gate_dict[gate_layer]-remove_dic_count_new[gate_layer]))
        net = map_gate_to_convbn_vgg(net, gate_layer, remove_dic_new[gate_layer], device)
        logging.info(net)

        # profile the forward pass
        # print the results

        if args.smooth:
            logging.info('Finished removing channels in '+str(gate_layer)+', finetune for several epochs.')
            optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01,
                        momentum=0.9, weight_decay=5e-4)
            scheduler = MultiStepLR(optimizer2, milestones=[5, 10], gamma=0.1)
            start_epoch = 0
            best_accuracy = finetune_and_evaluate(net, trainloader, testloader, optimizer2, scheduler, total_epochs=30, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
            logging.info('Best accuracy: {:.4f}'.format(best_accuracy))

    logging.info('Finished removing')
    
    flops_after_prune = flops(net, 32 if args.dataset == 'cifar10' else 224)
    logging.info("FLOPS: {} M".format(str(flops(net, 32 if args.dataset == 'cifar10' else 224))))
    logging.info("Params: {} M".format(str(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6)))
    logging.info("FLOPS pruned ratio: {:.4f}".format(1. - flops_after_prune / flops_base))

    if args.smooth:
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer2, milestones=[10, 15], gamma=0.2)
        if args.cosine:
            scheduler = CosineAnnealingLR(optimizer2, 5, 30, 5, 0.0)
        start_epoch = 0
        best_accuracy = finetune_and_evaluate(net, trainloader, testloader, optimizer2, scheduler, total_epochs=45, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
        logging.info('Best accuracy: {:.4f}'.format(best_accuracy))
    else:
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer2, milestones=[60, 120], gamma=0.2)
        if args.cosine:
            scheduler = CosineAnnealingLR(optimizer2, 5, 30, 5, 0.0)
        start_epoch = 0
        best_accuracy = finetune_and_evaluate(net, trainloader, testloader, optimizer2, scheduler, total_epochs=args.epochs, start_epoch=start_epoch, name=args.net, ratio=args.p, smooth=args.smooth)
        logging.info('Best accuracy: {:.4f}'.format(best_accuracy))
