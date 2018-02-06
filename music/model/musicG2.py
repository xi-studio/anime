import torch.nn as nn
import torch
from torch.nn import DataParallel

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator,self).__init__()
        self.conv1  = nn.Sequential(
            nn.Conv1d(input_nc, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2  = nn.Sequential(
            nn.Conv1d(ngf * 1, ngf * 2, 4, 2, 1),
            nn.Conv1d(ngf * 2, ngf * 4, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3  = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.Conv1d(ngf * 4, ngf * 4, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4  = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.Conv1d(ngf * 4, ngf * 4, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv5  = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.Conv1d(ngf * 4, ngf * 4, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv6  = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.Conv1d(ngf * 4, ngf * 4, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv7  = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.Conv1d(ngf * 4, ngf * 4, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv8  = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.Conv1d(ngf * 4, ngf * 4, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv9  = nn.Sequential(
            nn.Conv1d(ngf * 4, ngf * 8, 4, 2, 1),
            nn.Conv1d(ngf * 8, ngf * 8, 2),
            nn.LeakyReLU(0.2, True),
        )
        self.conv10  = nn.Sequential(
            nn.Conv1d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.Conv1d(ngf * 8, ngf * 8, 2),
            nn.LeakyReLU(0.2, True),
        )
                      

        self.dconv1  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 8, ngf * 8, 2),
            nn.ConvTranspose1d(ngf * 8, ngf * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv2  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 8 * 2, ngf * 8, 2),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv3  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4 * 2, ngf * 4, 2),
            nn.ConvTranspose1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv4  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4 * 2, ngf * 4, 2),
            nn.ConvTranspose1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv5  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4 * 2, ngf * 4, 2),
            nn.ConvTranspose1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv6  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4 * 2, ngf * 4, 2),
            nn.ConvTranspose1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv7  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4 * 2, ngf * 4, 2),
            nn.ConvTranspose1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv8  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4 * 2, ngf * 4, 2),
            nn.ConvTranspose1d(ngf * 4, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv9  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 4 * 2, ngf * 2, 2),
            nn.ConvTranspose1d(ngf * 2, ngf * 1, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.dconv10  = nn.Sequential(
            nn.ConvTranspose1d(ngf * 1 * 2, output_nc, 4, 2, 1),
        )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        e1  = self.conv1(input)
        e2  = self.conv2(e1)
        e3  = self.conv3(e2)
        e4  = self.conv4(e3)
        e5  = self.conv5(e4)
        e6  = self.conv6(e5)
        e7  = self.conv7(e6)
        e8  = self.conv8(e7)
        e9  = self.conv9(e8)
        e10 = self.conv10(e9)

        # Decoder
        d1_ = self.dropout(self.dconv1(e10))
        d1  = torch.cat((d1_, e9), 1)
        d2_ = self.dropout(self.dconv2(d1))
        d2  = torch.cat((d2_, e8), 1)
        d3_ = self.dropout(self.dconv3(d2))
        d3  = torch.cat((d3_, e7), 1)
        d4_ = self.dconv4(d3)
        d4  = torch.cat((d4_, e6), 1)
        d5_ = self.dconv5(d4)
        d5  = torch.cat((d5_, e5), 1)
        d6_ = self.dconv6(d5)
        d6  = torch.cat((d6_, e4), 1)
        d7_ = self.dconv7(d6)
        d7  = torch.cat((d7_, e3), 1)
        d8_ = self.dconv8(d7)
        d8  = torch.cat((d8_, e2), 1)
        d9_ = self.dconv9(d8)
        d9  = torch.cat((d9_, e1), 1)
        d10 = self.dconv10(d9)
        output = self.tanh(d10)
        return output
