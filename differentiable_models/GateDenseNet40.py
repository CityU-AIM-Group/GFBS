import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(torch.nn.Module):
    def __init__(self, out_planes):
        super(Gate, self).__init__()
        self.gate = nn.Parameter(torch.ones(1, out_planes, 1, 1), requires_grad=False)

    def forward(self, x):
        return self.gate * x

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.gate1 = Gate(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(self.relu(self.gate1(self.bn1(x))))
        # print(out.shape[1])
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.gate1 = Gate(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.avgpool1 = nn.AvgPool2d(2)
    def forward(self, x):
        out = self.avgpool1(self.conv1(self.gate1(self.relu(self.bn1(x)))))
        return out

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class GateDenseNet40(nn.Module):
    def __init__(self, depth=40, num_classes=10, growth_rate=12,
                 reduction=1.0, bottleneck=False):
        super(GateDenseNet40, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)))
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)))
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.gate1 = Gate(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.gate1(self.bn1(out)))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

def get_channels_densenet40(net):
    t = 0
    all_ = 0
    gate_remain_dict = {}
    channel_mapper_for_flops = []
    for name, params in net.named_parameters():
        if 'gate' in name: # select weights
            remain = torch.sum(params)
            pruned = len(params.squeeze()) - remain
            t+=pruned
            all_+=remain
            print(name, remain)
            gate_remain_dict[name] = int(remain.data.item())

            if channel_mapper_for_flops == []:
                channel_mapper_for_flops.append([3, int(remain.data.item())]) # [3, first channel]
            else:
                # print(t)
                channel_mapper_for_flops.append([channel_mapper_for_flops[-1][1], int(remain.data.item())]) # [last channel out, this channel]
    
    maxpool_inchannel_list = []
    for name, m in net.named_modules():
        if m.__class__.__name__ == 'MaxPool2d': # module.features.43 ==> module.features.41.gate
            l = name.split('.')
            l.append('gate')
            l[-2] = str(int(l[-2]) - 2) # The gate index is two layers before the maxpool layer (with a relu after)
            gate_name = '.'.join(l)
            maxpool_inchannel_list.append(gate_remain_dict[gate_name])
    if maxpool_inchannel_list != []:
        maxpool_inchannel_list.append(maxpool_inchannel_list[-1]) # For the average pooling

    print("pruned/total: ", t, all_)
    print("channel_mapper_for_flops: ", channel_mapper_for_flops)
    print("input channel before MaxPool2d layer (For VGG/DenseNet): ", maxpool_inchannel_list)
    return channel_mapper_for_flops

if __name__ == "__main__":
    net = GateDenseNet40()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(net)
    print(y.shape)
    get_channels_densenet40(net)