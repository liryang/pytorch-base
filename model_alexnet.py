import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.inputsize = (224, 224)
        self.name = 'alexnet'
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def keep_weight(self):
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier[1].parameters():
            param.requires_grad = False
        for param in self.classifier[4].parameters():
            param.requires_grad = False


def alexnet(num_classes, pthpath=None, first=False):
    if first:
        model = AlexNet()
        model.load_state_dict(torch.load('F:/pratice/checkpoint/alexnet-owt-4df8aa71.pth'))
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        nn.init.xavier_normal_(model.classifier[6].weight.data)
        nn.init.constant_(model.classifier[6].bias.data, 0.0)
    else:
        model = AlexNet(num_classes=num_classes)
        if pthpath is not None:
            path = './checkpoint/' + model.name + '_' + pthpath
            model.load_state_dict(torch.load(path))

    return model


