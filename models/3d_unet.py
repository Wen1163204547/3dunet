import torch
from torch import nn

class UNet_3d(nn.Module):
    def __init__(self):
        super(UNet_3d, self).__init__()

        forw_chs = [1, 16, 32, 64, 128]
        back_chs = [128, 64, 32, 16]
        for i in xrange(len(forw_chs) - 1):
            block = nn.Sequential(
                    nn.Conv3d(forw_chs[i], forw_chs[i+1], (3, 3, 3), padding=1),
                    nn.BatchNorm3d(forw_chs[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(forw_chs[i+1], forw_chs[i+1], (3, 3, 3), padding=1),
                    nn.BatchNorm3d(forw_chs[i+1]),
                    nn.ReLU(inplace=True))
            setattr(self, 'encoder'+str(i+1), block)
        for i in xrange(len(back_chs) - 1):
            block = nn.Sequential(
                    nn.Conv3d(back_chs[i], back_chs[i+1], (3, 3, 3), padding=1),
                    nn.BatchNorm3d(back_chs[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(back_chs[i+1], back_chs[i+1], (3, 3, 3), padding=1),
                    nn.BatchNorm3d(back_chs[i+1]),
                    nn.ReLU(inplace=True))
            setattr(self, 'decoder'+str(i+1), block)
        self.decoder3 = nn.Sequential(
                    nn.Conv3d(32, 16, (3, 3, 3), padding=1),
                    nn.BatchNorm3d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(16, 16, (3, 3, 3), padding=1),
                    nn.BatchNorm3d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(16, 2, (1, 1, 1), padding=0))
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.upconv = nn.Upsample(scale_factor=(2, 2))
        #self.upconv1 = nn.ConvTranspose3d(512, 256, 2, stride=2, padding=0)
        self.upconv1 = nn.ConvTranspose3d(128, 64, 2, stride=2, padding=0)
        self.upconv2 = nn.ConvTranspose3d(64, 32, 2, stride=2, padding=0)
        self.upconv3 = nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0)

    def forward(self, x):
        out1 = self.encoder1(x)
        out2 = self.encoder2(self.maxpool(out1))
        out3 = self.encoder3(self.maxpool(out2))
        out4 = self.encoder4(self.maxpool(out3))
        out5 = self.decoder1(torch.cat([out3, self.upconv1(out4)], 1))
        out6 = self.decoder2(torch.cat([out2, self.upconv2(out5)], 1))
        out7 = self.decoder3(torch.cat([out1, self.upconv3(out6)], 1))
        return out7

def get_model():
    net = UNet_3d()
    return net
    
