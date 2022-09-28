import torch
import torch.nn as nn
import math

def autopad(k, p=None):
    if p is None:
        p = k // 2
    return p 

class ConvOne(nn.Module):

    def __init__(self, c1, c2, k, s):
        super(ConvOne, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, groups=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class ConvBasic(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(ConvBasic, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p), groups=g, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DepthConv(ConvBasic):

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super(DepthConv, self).__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class DepthwiseSeparable(nn.Module):

    def __init__(self, nin, nout, e=2):
        super(DepthwiseSeparable, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * e, kernel_size=3, padding=1, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin * e, nout, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BottleNeck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(BottleNeck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = DepthwiseSeparable(c1, c_)
        self.conv2 = DepthwiseSeparable(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return  x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class CSPBottleNeck(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(CSPBottleNeck, self).__init__()
        self.conv = ConvBasic(c1=c1, c2=c2//2, k=3, s=2, p=1)
        c1 = c2//2
        c_ = int(c2 * 0.5)
        self.conv1 = DepthwiseSeparable(c1, c_) 
        self.conv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.conv4 = DepthwiseSeparable(2 * c_, c2)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(BottleNeck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        
    def forward(self, x):
        x = self.conv(x)
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat([y1, y2], 1))))

class BackBone(nn.Module):

    def __init__(self, con, inc=3):
        super(BackBone, self).__init__()
        self.block1 = CSPBottleNeck(inc, con[0][0], con[0][1])
        self.block2 = CSPBottleNeck(con[0][0], con[1][0], con[1][1])
        self.block3 = CSPBottleNeck(con[1][0], con[2][0], con[2][1])
        self.block4 = CSPBottleNeck(con[2][0], con[3][0], con[3][1])
        self.block5 = CSPBottleNeck(con[3][0], con[4][0], con[4][1])
        self.block6 = CSPBottleNeck(con[4][0], con[5][0], con[5][1])
        self.block7 = CSPBottleNeck(con[5][0], con[6][0], con[6][1])

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        f = []
        f.append(x7)
        f.append(x6)
        f.append(x5)
        f.append(x4)
        f.append(x3)
        return f

class UpSample(nn.Module):
    
    def __init__(self,):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        return self.up(x)

class SPP(nn.Module):
    
    def __init__(self, c1, c2, K=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = 32
        self.conv1 = ConvBasic(c1, c_, 1, 1)
        self.conv2 = ConvBasic(c_ * (len(K) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in K])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))

class FPN(nn.Module):

    def __init__(self, pool_config):
        super(FPN, self).__init__()
        self.conv1 = ConvOne(pool_config[0], pool_config[0], 1, 1)
        self.conv2 = ConvOne(pool_config[1], pool_config[1], 1, 1)
        self.conv3 = ConvOne(pool_config[2], pool_config[2], 1, 1)
        self.conv4 = ConvOne(pool_config[3], pool_config[3], 1, 1)
        self.conv5 = ConvOne(pool_config[4], pool_config[4], 1, 1)
        self.conv6 = ConvOne(pool_config[2], pool_config[2], 1, 1)
        self.conv7 = ConvOne(pool_config[2], pool_config[2], 1, 1)
        self.up1 = UpSample()
        self.up2 = UpSample()
        self.up3 = UpSample()
        self.up4 = UpSample()
        self.up5 = UpSample()
        self.up6 = UpSample()
        self.up7 = UpSample()
        self.pool1 = SPP(1024+512, 512)
        self.pool2 = SPP(512+256, 512)
        self.pool3 = SPP(512+128, 256)
        self.pool4 = SPP(256+64, 256)

    def forward(self, f):
        x = self.conv1(f[0])
        x = self.up1(x)
        x1 = self.conv2(f[1])
        x = torch.cat([x, x1], 1)
        x = self.pool1(x)
        x = self.up2(x)
        x1 = self.conv3(f[2])
        x = torch.cat([x, x1], 1)
        x = self.pool2(x)
        x = self.up3(x)
        x1 = self.conv4(f[3])
        x = torch.cat([x, x1], 1)
        x = self.pool3(x)
        x = self.up4(x)
        x1 = self.conv5(f[4])
        x = torch.cat([x, x1], 1)
        x = self.pool4(x)
        x = self.conv6(x)
        x = self.up5(x)
        x = self.conv7(x)
        x = self.up6(x)
        return x

class SE(nn.Module):

    def __init__(self, c1, c2):
        super(SE, self).__init__()
        self.sq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c2, 1), 
            nn.SiLU(), 
            nn.Conv2d(c2, c1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + self.sq(x)

class InvertedResidualConv(nn.Module):

    def __init__(self, c1, c2, c3, k, s, p):
        super(InvertedResidualConv, self).__init__()
        self.residual = c1 == c2 and s == 1
        self.conv = nn.Sequential(
            ConvBasic(c1=c1, c2=c2, k=k, s=s), 
            SE(c2, c3), 
            nn.Conv2d(c2, c2, 1) 
        )

    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Head(nn.Module):

    ### use inverted residual block here
    ## output gives 
    def __init__(self, num_class=100, box=6, c=512):
        super(Head, self).__init__()
        c2 = c * 2
        self.c1 = InvertedResidualConv(c, c, c2, 1, 1, 1)
        self.c2 = InvertedResidualConv(c, (num_class + 5), c2, 1, 1, 1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x

class ObjectDetectionUnet(nn.Module):

    def __init__(self, model_config, pool_config):
        super(ObjectDetectionUnet, self).__init__()
        self.back = BackBone(model_config)
        self.fpn = FPN(pool_config)
        self.head = Head(605, 5, 256)

    def forward(self, x):
        x = self.back(x)
        x = self.fpn(x)
        x = self.head(x)
        return x

def get_unet_model():
    model_config = [
        (16, 2),
        (32, 2),
        (64, 2), 
        (128, 4), 
        (256, 4), 
        (512, 4), 
        (1024, 4)
    ]
    pool_config = [
        1024, 
        512,
        256, 
        128, 
        64,
    ]
    model = ObjectDetectionUnet(model_config, pool_config)
    return model
