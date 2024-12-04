import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.half = self.in_channel != self.out_channel

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2 if self.half else 1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        if self.half:
            self.half_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        tool = x.clone()
        if self.half:
            tool = self.half_conv(tool)
        result = self.model(x)
        result += tool
        result = torch.relu_(result)
        return result

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=out_channel)
        )
    
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                            # 若是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out',   # 用（何）kaiming_normal_法初始化权重
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                    # 初始化偏重为0
            elif isinstance(m, nn.Linear):            # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)    # 正态分布初始化
                nn.init.constant_(m.bias, 0)          # 初始化偏重为0

    def forward(self,x):
        return self.model(x)


        

