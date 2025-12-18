import torch 
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=48, base_ch=64):
        super().__init__()

        # Down block
        self.down1 = self.block(in_ch, base_ch)
        self.down2 = self.block(base_ch, base_ch*2)
        self.down3 = self.block(base_ch*2, base_ch*4)

        # Mid block 
        self.mid = self.block(base_ch*4, base_ch*4)

        # Up block
        self.up3 = self.block(base_ch*8, base_ch*2)
        self.up2 = self.block(base_ch*4, base_ch)
        self.up1 = self.block(base_ch*2, base_ch)

        # Head
        self.head = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(F.avg_pool2d(d1, 2))
        d3 = self.down3(F.avg_pool2d(d2, 2))

        m = self.mid(F.avg_pool2d(d3, 2))

        u3 = F.interpolate(m, scale_factor=2)
        u3 = self.up3(torch.cat([u3, d3], dim=1))

        u2 = F.interpolate(u3, scale_factor=2)
        u2 = self.up2(torch.cat([u2, d2], dim=1))

        u1 = F.interpolate(u2, scale_factor=2)
        u1 = self.up1(torch.cat([u1, d1], dim=1))

        logits = self.head(u1)  # [B, 48, 32, 32]
        return logits
