import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class ResidualConv(nn.Module):
    def __init__(self, input_channel, output_channel, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=7, stride=stride, padding=padding),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=7, stride=1, padding=padding),
            nn.BatchNorm1d(output_channel),
            nn.ReLU()
        )
        self.conv_skip = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm1d(output_channel),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose1d(input_channel, output_channel, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)


class ResUnet(nn.Module):
    def __init__(self, channel=1, filters=[8, 16, 32, 64]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(channel, filters[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(),
            nn.Conv1d(filters[0], filters[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(),
        )
        self.input_skip = nn.Sequential(
            nn.Conv1d(channel, filters[0], kernel_size=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 3)

        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 3)


        self.bridge = ResidualConv(filters[2], filters[3], 2, 3)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 3)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 3)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 3)


        self.output_layer = nn.Sequential(
            nn.Conv1d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # Encoder
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.residual_conv_1(x1)

        x3 = self.residual_conv_2(x2)

        # Bridge
        x4 = self.bridge(x3)
        # Decoder
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        output = output.squeeze()
        return output


class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.c = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm1d(64)
        self.a = nn.ReLU()

    def forward(self, x):
        x = self.c(x)
        x = self.BN(x)
        x = self.a(x)
        return x

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.input = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.block1 = block()
        self.block2 = block()
        self.block3 = block()
        self.fc = nn.Linear(in_features=64*1280, out_features=1280)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x.view(x.shape[0], -1))
        x = x.squeeze()
        return x

class RNN_lstm(nn.Module):
    def __init__(self):
        super(RNN_lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, batch_first=True)  # 一层lstm， 第一维是batch

        self.block1 = nn.Sequential(nn.Linear(in_features=1280, out_features=1280),
                                    nn.ReLU(),
                                    nn.Dropout(0.3))

        self.block2 = nn.Sequential(nn.Linear(in_features=1280, out_features=1280),
                                    nn.ReLU(),
                                    nn.Dropout(0.3))
        self.fc = nn.Linear(in_features=1280, out_features=1280)

    def forward(self, x):

        x = x.unsqueeze(2)  # (batch, 1280, 1)
        x, (ht, ct) = self.lstm(x)

        x = self.block1(x.reshape(x.shape[0], -1))
        x = self.block2(x)
        x = self.fc(x)
        # x = x.squeeze()
        return x

class basic_block(nn.Module):
    def __init__(self, kernel_size):
        super(basic_block, self).__init__()
        self.c1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.c2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.c3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.BN1 = nn.BatchNorm1d(num_features=32)
        self.BN2 = nn.BatchNorm1d(num_features=16)
        self.BN3 = nn.BatchNorm1d(num_features=32)
        self.a = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.BN1(x)
        x = self.a(x)

        x = self.c2(x)
        x = self.BN2(x)
        x = self.a(x)

        x = self.c3(x)
        x = self.BN3(x)
        x = self.a(x)

        return x

class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.basic1 = nn.Sequential(
            basic_block(kernel_size=3),
            basic_block(kernel_size=3)
        )
        self.basic2 = nn.Sequential(
            basic_block(kernel_size=5),
            basic_block(kernel_size=5)
        )
        self.basic3 = nn.Sequential(
            basic_block(kernel_size=7),
            basic_block(kernel_size=7)
        )

    def forward(self, x):
        x1 = self.basic1(x)
        x2 = self.basic2(x)
        x3 = self.basic3(x)

        y = torch.cat([x1, x2, x3], dim=-2)
        return y

class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.c2 = nn.Conv1d(in_channels=32*3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.BN1 = nn.BatchNorm1d(num_features=32)
        self.BN2 = nn.BatchNorm1d(num_features=32)
        self.a = nn.ReLU()
        self.fc = nn.Linear(in_features=32*1280, out_features=1280)
        self.block1 = block()


    def forward(self, input):
        input = input.unsqueeze(1)
        x = self.c1(input)
        x = self.BN1(x)
        x = self.a(x)

        x = self.block1(x)

        x = self.c2(x)
        x = self.BN2(x)
        x = self.a(x)
        x = self.fc(x.view(x.shape[0], -1))

        return x


if __name__ == '__main__':
    model = SCNN()
    Unet = ResUnet(1)
    # summary(Unet, input_size=(1280,), batch_size=1000)
    x = torch.randn((1000, 1280))
    y = Unet(x)
    print(y.shape)