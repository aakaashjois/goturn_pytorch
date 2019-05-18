import pickle
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, pretrained=None):
        super(Model, self).__init__()
        self.features_previous = nn.Sequential()
        self.features_previous.add_module('conv11', nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4))
        self.features_previous.add_module('relu11', nn.ReLU())
        self.features_previous.add_module('pool11', nn.MaxPool2d(kernel_size=3, stride=2))
        self.features_previous.add_module('lrn11', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75))
        self.features_previous.add_module('conv12',
                                          nn.Conv2d(in_channels=96, out_channels=256, padding=2, kernel_size=5,
                                                    groups=2))
        self.features_previous.add_module('relu12', nn.ReLU())
        self.features_previous.add_module('pool12', nn.MaxPool2d(kernel_size=3, stride=2))
        self.features_previous.add_module('lrn12', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75))
        self.features_previous.add_module('conv13',
                                          nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3))
        self.features_previous.add_module('relu13', nn.ReLU())
        self.features_previous.add_module('conv14',
                                          nn.Conv2d(in_channels=384, out_channels=384, padding=1, kernel_size=3,
                                                    groups=2))
        self.features_previous.add_module('relu14', nn.ReLU())
        self.features_previous.add_module('conv15',
                                          nn.Conv2d(in_channels=384, out_channels=256, padding=1, kernel_size=3,
                                                    groups=2))
        self.features_previous.add_module('relu15', nn.ReLU())
        self.features_previous.add_module('pool15', nn.MaxPool2d(kernel_size=3, stride=2))

        self.features_current = nn.Sequential()
        self.features_current.add_module('conv21', nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4))
        self.features_current.add_module('relu21', nn.ReLU())
        self.features_current.add_module('pool21', nn.MaxPool2d(kernel_size=3, stride=2))
        self.features_current.add_module('lrn21', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75))
        self.features_current.add_module('conv22',
                                         nn.Conv2d(in_channels=96, out_channels=256, padding=2, kernel_size=5,
                                                   groups=2))
        self.features_current.add_module('relu22', nn.ReLU())
        self.features_current.add_module('pool22', nn.MaxPool2d(kernel_size=3, stride=2))
        self.features_current.add_module('lrn22', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75))
        self.features_current.add_module('conv23',
                                         nn.Conv2d(in_channels=256, out_channels=384, padding=1, kernel_size=3))
        self.features_current.add_module('relu23', nn.ReLU())
        self.features_current.add_module('conv24',
                                         nn.Conv2d(in_channels=384, out_channels=384, padding=1, kernel_size=3,
                                                   groups=2))
        self.features_current.add_module('relu24', nn.ReLU())
        self.features_current.add_module('conv25',
                                         nn.Conv2d(in_channels=384, out_channels=256, padding=1, kernel_size=3,
                                                   groups=2))
        self.features_current.add_module('relu25', nn.ReLU())
        self.features_current.add_module('pool25', nn.MaxPool2d(kernel_size=3, stride=2))

        self.regressor = nn.Sequential()
        self.regressor.add_module('fc6', nn.Linear(in_features=18432, out_features=4096))
        self.regressor.add_module('relu6', nn.ReLU())
        self.regressor.add_module('drop6', nn.Dropout())
        self.regressor.add_module('fc7', nn.Linear(in_features=4096, out_features=4096))
        self.regressor.add_module('relu7', nn.ReLU())
        self.regressor.add_module('drop7', nn.Dropout())
        self.regressor.add_module('fc8', nn.Linear(in_features=4096, out_features=4))

        if pretrained:
            model_weights = pickle.load(open(pretrained, 'rb'))
            for name, module in self.features_previous.named_modules():
                if name in model_weights.keys():
                    module.weight = nn.Parameter(torch.from_numpy(model_weights[name]['weight']))
                    module.bias = nn.Parameter(torch.from_numpy(model_weights[name]['bias']))
            for name, module in self.features_current.named_modules():
                if name in model_weights.keys():
                    module.weight = nn.Parameter(torch.from_numpy(model_weights[name]['weight']))
                    module.bias = nn.Parameter(torch.from_numpy(model_weights[name]['bias']))
            for name, module in self.regressor.named_modules():
                if name in model_weights.keys():
                    module.weight = nn.Parameter(torch.from_numpy(model_weights[name]['weight']))
                    module.bias = nn.Parameter(torch.from_numpy(model_weights[name]['bias']))

    def forward(self, previous, current):
        out_previous = self.features_previous(previous)
        out_current = self.features_current(current)
        concat = torch.cat((out_previous.view(out_previous.size()[0], -1),
                            out_current.view(out_current.size()[0], -1)), dim=1)
        output = self.regressor(concat)
        return torch.mul(output, 10)
