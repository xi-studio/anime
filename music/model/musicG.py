#Credit: code copied from https://github.com/mrzhu-cool/pix2pix-pytorch/blob/master/models.py
import torch.nn as nn
import torch
from torch.nn import DataParallel

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        self.conv1 = nn.Conv1d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv1d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv1d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv1d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv1d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv1d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv1d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv1d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose1d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose1d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose1d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose1d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose1d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose1d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose1d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose1d(ngf * 2, output_nc, 4, 2, 1)


        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        e1 = self.conv1(input)
        e2 = self.conv2(self.leaky_relu(e1))
        # state size is (ngf x 2) x 64 x 64
        e3 = self.conv3(self.leaky_relu(e2))
        # state size is (ngf x 4) x 32 x 32
        e4 = self.conv4(self.leaky_relu(e3))
        # state size is (ngf x 8) x 16 x 16
        e5 = self.conv5(self.leaky_relu(e4))
        # state size is (ngf x 8) x 8 x 8
        e6 = self.conv6(self.leaky_relu(e5))
        # state size is (ngf x 8) x 4 x 4
        e7 = self.conv7(self.leaky_relu(e6))
        # state size is (ngf x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1_ = self.dropout(self.dconv1(self.relu(e8)))
        # state size is (ngf x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.dconv2(self.relu(d1)))
        # state size is (ngf x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.dconv3(self.relu(d2)))
        # state size is (ngf x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.dconv4(self.relu(d3))
        # state size is (ngf x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.dconv5(self.relu(d4))
        # state size is (ngf x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.dconv6(self.relu(d5))
        # state size is (ngf x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.dconv7(self.relu(d6))
        # state size is (ngf) x 128 x 128
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.relu(d7))
        # state size is (nc) x 256 x 256
        output = self.tanh(d8)
        return output
