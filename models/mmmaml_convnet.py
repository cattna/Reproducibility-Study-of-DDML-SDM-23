import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np


class MContextNet3(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size, classify=False):
        super(MContextNet3, self).__init__()

        # Keep same dimensions
        padding = (kernel_size - 1) // 2

        self.context_net = nn.Sequential(
                                nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=padding),
                                nn.BatchNorm2d(hidden_dim),
                                nn.ReLU(),
#                                 nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
#                                 nn.BatchNorm2d(hidden_dim),
#                                 nn.ReLU(),
                                nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=padding)
                            )

    def forward(self, x):
        out = self.context_net(x)
        return out

class MResNet3(nn.Module):

    def __init__(self, num_channels, num_classes, model_name, pretrained=None,
                 avgpool=False):
        super(MResNet3, self).__init__()

        self.model = torchvision.models.__dict__[model_name](pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Change number of input channels from 3 to whatever is needed
        # to take in the context also.
        if num_channels != 3:
            model_inplanes = 64
            old_weights = self.model.conv1.weight.data
            self.model.conv1 = nn.Conv2d(num_channels, model_inplanes,
                             kernel_size=7, stride=2, padding=3, bias=False)

            if pretrained:
                for i in range(num_channels):
                    self.model.conv1.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]

        if avgpool:
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, adap_params=None):
        if adap_params == None:
            out = self.model(x)
        else:
            out = self.model(x)
        return out


class MMAMLConvNet3(nn.Module):

    def __init__(self, num_channels=3, prediction_net='resnet18', num_classes=10,
                 support_size=50, use_context=None, n_context_channels=None,
                 pretrained=None, context_num=2, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.support_size = support_size
        self.use_context = use_context
        self.context_num = context_num
        self.n_context_channels = n_context_channels or num_channels

        if use_context:
            self.context_net = nn.ModuleList([
                MContextNet3(num_channels, self.n_context_channels, hidden_dim=64, kernel_size=5)
                for _ in range(context_num)
            ])
            pred_channels = num_channels + self.n_context_channels
        else:
            pred_channels = num_channels

        if prediction_net == 'convnet':
            self.prediction_net = MConvNet3(num_channels=pred_channels, num_classes=num_classes,
                                            cml=bool(use_context))
        else:  # ResNet
            self.prediction_net = MResNet3(num_channels=pred_channels, num_classes=num_classes,
                                           model_name=prediction_net, pretrained=pretrained)
        #self.prediction_net = nn.DataParallel(self.prediction_net)

    def get_features(self, x):
        if self.use_context:
            b, c, h, w = x.shape
            meta_bs = b // self.support_size if b % self.support_size == 0 else b
            sb = b // self.context_num

            contexts = []
            for i in range(self.context_num):
                ctx = self.context_net[i](x[i*sb:(i+1)*sb])
                ctx = ctx.view(meta_bs//self.context_num, -1, self.n_context_channels, h, w).mean(dim=1)
                ctx = ctx.repeat_interleave(self.support_size, dim=0)
                contexts.append(ctx)
            x = torch.cat([x, torch.cat(contexts, dim=0)], dim=1)

        net = self.prediction_net
        if hasattr(net, "model"):  # ResNet
            m = net.model
            x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
            x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1)

        # ConvNet
        x = net.conv1(x) if net.cml else net.conv0(x)
        if not net.cml: x = net.conv1(x)
        x = net.conv2(x)
        return net.adaptive_pool(x).flatten(1)


    def forward(self, x):
        if self.use_context:
            b, c, h, w = x.shape
            meta_bs = b // self.support_size if b % self.support_size == 0 else b
            sb = b // self.context_num

            contexts = []
            for i in range(self.context_num):
                ctx = self.context_net[i](x[i*sb:(i+1)*sb])
                ctx = ctx.view(meta_bs//self.context_num, -1, self.n_context_channels, h, w).mean(dim=1)
                ctx = ctx.repeat_interleave(self.support_size, dim=0)
                contexts.append(ctx)
            x = torch.cat([x, torch.cat(contexts, dim=0)], dim=1)

        return self.prediction_net(x)

class MConvNet3(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, cml=True, **kwargs):
        super(MConvNet3, self).__init__()

        hidden_dim = 128
        kernel_size = 5

        self.cml = cml
        padding = (kernel_size - 1) // 2
        if cml:
            self.conv1 = nn.Sequential(
                            nn.Conv2d(num_channels, hidden_dim, kernel_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )
        else:
            print("using larger model")
            self.conv0 = nn.Sequential(
                            nn.Conv2d(num_channels, hidden_dim, kernel_size, padding=padding),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                        )

            self.conv1 = nn.Sequential(
                            nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2)
                        )


        self.conv2 = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, kernel_size),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        self.final = nn.Sequential(
                    nn.Linear(hidden_dim, 200),
                    nn.ReLU(),
                    nn.Linear(200, num_classes)
                  )


    def forward(self, x):
        """Returns logit with shape (batch_size, num_classes)"""

        # x shape: batch_size, num_channels, w, h

        if self.cml:
            out = self.conv1(x)
        else:
            out = self.conv0(x)
            out = self.conv1(out)
        out = self.conv2(out)
        out = self.adaptive_pool(out).squeeze()
        out = self.final(out)

        return out

    def args_dict(self):
        """
            Model args used when saving and loading the model from a ckpt
        """

        #model_args = {
        #        'hidden_dim': self.hidden_dim,
        #        'num_classes': self.num_classes,
        #        'kernel_size': self.kernel_size,
        #        'num_channels': self.num_channels
        #        }
        model_args = {}

        return model_args


    
