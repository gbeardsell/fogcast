import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualBlock2d(nn.Module):
    def __init__(self,in_channels: int,out_channels: int, stride: int=1):
        super().__init__()
        if stride not in (1, 2) or out_channels % 8:
            raise ValueError("stride must be 1/2 and output channels divisible by eight")
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels, eps=1e-5),
            nn.SiLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels, eps=1e-5)
        )
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                    nn.AvgPool2d(2, stride=2, padding=0, ceil_mode=True) if stride == 2 else nn.Identity(),
                    nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                    nn.GroupNorm(8, out_channels, eps=1e-5),
            )
        self.activation = nn.SiLU(inplace=False)

        def forward(self, x:Tensor) -> Tensor:
            return self.activation(self.main(x) + self.shortcut(x))

class ResidualUNet(nn.Module):
    model_version = "fogcast_nbm_resunet_v2"
    fine_channels = 5
    coarse_channels = 25
    horizons = 10

    def __init__(self):
        super().__init__()
        def block(cin, cout, stride):
            return ResidualBlock2d(cin, cout, stride)
        self.stem = block(5,16,1)
        self.down1 = block(16,32,2)
        self.down2 = block(32,32,2)
        self.fuse = block(57,64,1)
        self.down3 = block(64,128,2)
        self.bottleneck = block(128,256,2)
        self.dec3 = block(384, 128, 1)
        self.dec2 = block(128,64,1)
        self.dec1 = block(64,32,1)
        self.dec0 = block(32,16,1)
        self.head = nn.Conv2d(16,10,1,bias=True)

        @staticmethod
        def _decode(x: Tensor, skip: Tensor, block: nn.Module) -> Tensor:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return block(torch.cat((x, skip), dim=1))

        def forward(self, x_fine: Tensor, x_coarse: Tensor) -> Tensor:
            if (x_fine.ndim != 4 or x_fine.shape[1:] != (5,400,400) 
                or x_coarse.ndim != 4 or x_coarse.shape[1:] != (25,400,400)
                or x_fine.shape[0] != x_coarse.shape[0]):
                raise ValueError("expected matched [B,5,400,400] and [B,25,100,100] inputs")
            if not bool(torch.isfinite(x_fine).all() and torch.isfinite(x_coarse).all()):
                raise ValueError("forecast inputs must be finite; invalid pixels require neutral fill and masks")
            e0 = self.stem(x_fine)
            e1 = self.down1(e0)
            pre = self.down2(e1)
            e2 = self.fuse(torch.cat((pre, x_coarse), dim=1))
            e3 = self.down3(e2)
            b = self.bottleneck(e3)
            d3 = self._decode(b, e3, self.dec3)
            d2 = self._decode(d3, e2, self.dec2)
            d1 = self._decode(d2, e1, self.dec1)
            d0 = self._decode(d1, e0, self.dec0)
            return self.head(d0)


            
            





        
                    