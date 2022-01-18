import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import pdb
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        params = model_zoo.load_url(model_urls['densenet121'])
        params = {k: v for k, v in params.items() if k in model_dict}
        model_dict.update(params)
        model.load_state_dict(model_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        params = model_zoo.load_url(model_urls['densenet169'])
        params = {k: v for k, v in params.items() if k in model_dict}
        model_dict.update(params)
        model.load_state_dict(model_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(4, 12, 48, 32),
                     **kwargs)
    # if pretrained:
    #     model_dict = model.state_dict()
    #     params = model_zoo.load_url(model_urls['densenet201'])
    #     params = {k: v for k, v in params.items() if k in model_dict}
    #     model_dict.update(params)
    #     model.load_state_dict(model_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet161']))
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


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, fc_layer1=1024, fc_layer2=128, global_pooling_size=(8,4)):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        num_features_list = []
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            num_features_list.append(num_features)
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.layer_fc0 = nn.Sequential(
            nn.Linear(num_features_list[0], fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer_fc1 = nn.Sequential(
            nn.Linear(num_features_list[1], fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer_fc2 = nn.Sequential(
            nn.Linear(num_features_list[2], fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer_fc3 = nn.Sequential(
            nn.Linear(num_features_list[3], fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.avgpool0 = nn.Sequential(
            nn.BatchNorm2d(num_features_list[0]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d([x*2**(3-0) for x in global_pooling_size]),
        )
        self.avgpool1 = nn.Sequential(
            nn.BatchNorm2d(num_features_list[1]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d([x*2**(3-1) for x in global_pooling_size]),
        )
        self.avgpool2 = nn.Sequential(
            nn.BatchNorm2d(num_features_list[2]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d([x*2**(3-2) for x in global_pooling_size]),
        )
        self.avgpool3 = nn.AvgPool2d([x*1 for x in global_pooling_size])

        self.fusion_conv = nn.Conv1d(4,1,kernel_size=1, bias=False)


    def forward(self, x):
        x = x.view(-1, *x.size()[-3:])
        out_index = [4,6,8]
        out = []
        for index, feature in enumerate(self.features.children()):
            x = feature(x)
            if index in out_index:
                layer_index = out_index.index(index)
                temp = getattr(self, 'avgpool'+str(layer_index))(x)
                temp = temp.view(temp.size(0), -1)
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
            return x5
        out.append(x5)
        return out