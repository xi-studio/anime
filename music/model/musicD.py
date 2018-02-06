import torch.nn as nn
from torch.nn import DataParallel

class Discriminator(nn.Module):
    def __init__(self,input_nc,output_nc,ndf):
        super(Discriminator,self).__init__()
        self.conv  = nn.Sequential(
            nn.Conv1d(input_nc + output_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(ndf * 1, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(ndf * 4, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(ndf * 4, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(ndf * 4, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(ndf * 4, ndf * 4, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self,x):
        out = self.conv(x)
        return out
