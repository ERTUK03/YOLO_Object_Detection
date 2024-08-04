import torch
from torchvision.models import resnet50, ResNet50_Weights

class YOLO(torch.nn.Module):
    def __init__(self, resnet_backbone = True):
        super().__init__()
        layers = []
        if resnet_backbone:
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            backbone.requires_grad_(False)
            backbone.avgpool = torch.nn.Identity()
            backbone.fc = torch.nn.Identity()
            layers.extend(self.add_conv(in_channels = 2048, out_channels=1024, kernel_size=3))
            layers.extend(self.add_conv(in_channels = 1024, out_channels=1024, kernel_size=3, stride=2))
            layers.extend(self.add_conv(in_channels = 1024, out_channels=1024, kernel_size=3))
            layers.extend(self.add_conv(in_channels = 1024, out_channels=1024, kernel_size=3))
            layers.extend(self.add_head())
            self.Layers = torch.nn.Sequential(backbone,Reshape(2048,14,14), *layers)
        else:
            layers.extend(self.add_conv(in_channels = 3, out_channels=64, kernel_size=7, stride=2, padding = 3))
            layers.append(torch.nn.MaxPool2d(kernel_size=2))
            layers.extend(self.add_conv(in_channels = 64, out_channels=192, kernel_size=3))
            layers.append(torch.nn.MaxPool2d(kernel_size=2))
            layers.extend(self.add_conv(in_channels = 192, out_channels=128, kernel_size=1))
            layers.extend(self.add_conv(in_channels = 128, out_channels=256, kernel_size=3))
            layers.extend(self.add_conv(in_channels = 256, out_channels=256, kernel_size=1))
            layers.extend(self.add_conv(in_channels = 256, out_channels=512, kernel_size=3))
            layers.append(torch.nn.MaxPool2d(kernel_size=2))
            for _ in range(4):
                layers.extend(self.add_conv(in_channels = 512, out_channels=256, kernel_size=1))
                layers.extend(self.add_conv(in_channels = 256, out_channels=512, kernel_size=3))
            layers.extend(self.add_conv(in_channels = 512, out_channels=512, kernel_size=1))
            layers.extend(self.add_conv(in_channels = 512, out_channels=1024, kernel_size=3))
            layers.append(torch.nn.MaxPool2d(kernel_size=2))
            for _ in range(2):
                layers.extend(self.add_conv(in_channels = 1024, out_channels=512, kernel_size=1))
                layers.extend(self.add_conv(in_channels = 512, out_channels=1024, kernel_size=3))
            layers.extend(self.add_conv(in_channels = 1024, out_channels=1024, kernel_size=3))
            layers.extend(self.add_conv(in_channels = 1024, out_channels=1024, kernel_size=3, stride=2))
            layers.extend(self.add_conv(in_channels = 1024, out_channels=1024, kernel_size=3))
            layers.extend(self.add_conv(in_channels = 1024, out_channels=1024, kernel_size=3))
            layers.extend(self.add_head())

            self.Layers = torch.nn.Sequential(*layers)

    def add_conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        conv_block = [torch.nn.Conv2d(in_channels = in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=1 if kernel_size==3 else padding),
                      torch.nn.LeakyReLU(0.1),
                      torch.nn.BatchNorm2d(out_channels)]
        return conv_block

    def add_head(self):
        head = [torch.nn.Flatten(),
                torch.nn.Linear(7*7*1024, 4096),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(4096, 7*7*30),
                torch.nn.Sigmoid()]
        return head

    def forward(self, x):
        x = self.Layers(x)
        return x.view(-1, 7, 7, 30)

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return torch.reshape(x, (-1, *self.shape))
