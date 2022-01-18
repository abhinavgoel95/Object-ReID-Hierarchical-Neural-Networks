import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import pdb
from collections import OrderedDict

__all__ = ['DenseNet', 'downcolor_densenet201',]


def downcolor_densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=352, growth_rate=32, block_config=(3, 12, 48, 32),
                    **kwargs)
    return model

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm_2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, fc_layer1=1024, fc_layer2=128, global_pooling_size=(8,4)):
        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
        #    ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        num_features_list = []
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            num_features_list.append(num_features)
            break

        self.layer_fc0 = nn.Sequential(
            nn.Linear(num_features_list[0], fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.avgpool0 = nn.Sequential(
            nn.BatchNorm2d(num_features_list[0]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d([x*2**(3-3) for x in global_pooling_size]),
        )
        self.semantic_features = nn.Sequential(
            nn.Conv2d(num_features_list[0], 64, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding = 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8)
        )

        self.semantic_classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9),
        )

    def forward(self, x):
        x = x.view(-1, *x.size()[-3:])
        out_index = [3,6,8]
        out = []
        for index, feature in enumerate(self.features.children()):
            x = feature(x)
            if index in out_index:
                temp1 = self.semantic_features(x)
                layer_index = out_index.index(index)
                temp = getattr(self, 'avgpool'+str(layer_index))(x)
                temp = temp.view(temp.size(0), -1)
                temp1 = temp1.view(temp1.size(0), -1)
                temp1 = self.semantic_classifier(temp1)
                temp = getattr(self,'layer_fc'+str(layer_index))(temp)
                out.append(temp)
                break

        #temp = F.relu(x, inplace=True)
        #temp = self.avgpool3(temp).view(temp.size(0), -1)
        #temp = self.layer_fc3(temp)
        #out.append(temp)
        out_unsqueeze = [y.unsqueeze(dim=1) for y in out]
        x5 = torch.cat(out_unsqueeze, dim=1)
        #x5 = self.fusion_conv(x5)
        x5 = x5.view(x5.size(0),-1)
        if self.training is False:
            return x5, temp1, True, None
        out.append(x5)
        return x5, temp1
    
    def evaluate(self, x):
        x = x.view(-1, *x.size()[-3:])
        out_index = [3,6,8]
        out = []
        for index, feature in enumerate(self.features.children()):
            x = feature(x)
            if index in out_index:
                temp1 = self.semantic_features(x)
                layer_index = out_index.index(index)
                temp = getattr(self, 'avgpool'+str(layer_index))(x)
                temp = temp.view(temp.size(0), -1)
                temp1 = temp1.view(temp1.size(0), -1)
                temp1 = self.semantic_classifier(temp1)
                temp = getattr(self,'layer_fc'+str(layer_index))(temp)
                out.append(temp)
                break

        # #temp = F.relu(x, inplace=True)
        # #temp = self.avgpool3(temp).view(temp.size(0), -1)
        # #temp = self.layer_fc3(temp)
        # #out.append(temp)
        # out_unsqueeze = [y.unsqueeze(dim=1) for y in out]
        # x5 = torch.cat(out_unsqueeze, dim=1)
        # #x5 = self.fusion_conv(x5)
        # x5 = x5.view(x5.size(0),-1)
        # if self.training is False:
        #     return x5, temp1, True, None
        # out.append(x5)
        return x, temp1, temp