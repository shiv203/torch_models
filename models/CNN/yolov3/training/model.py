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

class BottleNeck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(BottleNeck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = DepthConv(c1, c_, 1, 1)
        self.conv2 = ConvBasic(c_, c2, 3, 1, g=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return  x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class CSPBottleNeck(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(CSPBottleNeck, self).__init__()
        c_ = int(c2 * 0.5)
        self.conv1 = ConvBasic(c1, c_, 1, 1) 
        self.conv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.conv4 = ConvBasic(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(BottleNeck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat([y1, y2], 1))))
      
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

class DarkNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1024, p=0.2):
        super(DarkNet, self).__init__()
        self.conv1 = ConvBasic(c1=in_ch, c2=16, k=3, s=1, p=1)
        self.conv2 = ConvBasic(c1=16, c2=32, k=3, s=2, p=1)
        self.blockcsp1 = CSPBottleNeck(c1=32, c2=64, n=1)
        self.conv3 = ConvBasic(c1=64, c2=64, k=3, s=2, p=1)
        self.blockcsp2 = CSPBottleNeck(c1=64, c2=128, n=2)
        self.conv4 = ConvBasic(c1=128, c2=128, k=3, s=2, p=1)
        self.blockcsp3 = CSPBottleNeck(c1=128, c2=256, n=8)
        self.conv5 = ConvBasic(c1=256, c2=256, k=3, s=2, p=1)
        self.blockcsp4 = CSPBottleNeck(c1=256, c2=512, n=8)
        self.conv6 = ConvBasic(c1=512, c2=512, k=3, s=2, p=1)
        self.blockcsp5 = CSPBottleNeck(c1=512, c2=1024, n=4)
        self.dr1 = nn.Dropout(p)
        self.dr2 = nn.Dropout(p)
        self.dr3 = nn.Dropout(p)
        self.dr4 = nn.Dropout(p)
        
       
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.blockcsp1(x)
        x = self.dr1(x)
        x = self.conv3(x)
        x = self.blockcsp2(x)
        x = self.dr2(x)
        x = self.conv4(x)
        x1 = self.blockcsp3(x)
        x = self.conv5(x1)
        x = self.dr3(x)
        x2 = self.blockcsp4(x)
        x = self.conv6(x2)
        x = self.dr4(x)
        x = self.blockcsp5(x)
        return x1, x2, x
 
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

class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        c_ = 32
        super(SPPF, self).__init__()
        self.conv1 = ConvBasic(c1, c_, 1, 1)
        self.conv2 = ConvBasic(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class FPN(nn.Module):

    def __init__(self):
        super(FPN, self).__init__()
        self.up1 = UpSample()
        self.up2 = UpSample()
        self.up3 = UpSample()
        self.pool1 = SPP(1024+512, 512)
        self.pool2 = SPP(1024+256+512, 512)

        self.conv1 = ConvBasic(1024, 1024, 1, 1) 
        self.conv2 = ConvBasic(512, 512, 1, 1)
        self.conv3 = ConvBasic(256, 256, 1, 1)
    
    def forward(self, x1, x2, x):
        ## x 16, 16 1024
        ## x2 32, 32 512
        ## x1 64 64 512
        x = self.conv1(x)
        x = self.up1(x)
        x2 = self.conv2(x2)
        x2 = torch.cat([x, x2], 1)
        x_ = self.pool1(x2)
        x2 = self.up2(x_)
        x1 = self.conv3(x1)
        x = self.up3(x)
        x1 = torch.cat([x1, x2, x], 1)
        x1 = self.pool2(x1)

        ## x1 64 512
        ## x_ 32 512



        ## conv layers 
        ## concat and pool
        return x1, x_

class Head(nn.Module):
    ### use inverted residual block here
    ## output gives 
    def __init__(self, num_class=100, box=6, c=512):
        super(Head, self).__init__()
        c2 = c // 2
        self.c1 = InvertedResidualConv(c, c, c2, 1, 1, 1)
        self.c2 = InvertedResidualConv(c, num_class, c2, 1, 1, 1)
        self.c3 = InvertedResidualConv(c, c, c2, 1, 1, 1)
        self.c4 = InvertedResidualConv(c, box, c2, 1, 1, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.c2(x1)
        x2 = self.c3(x)
        x2 = self.c4(x2)
        return x1, x2

class ObjectDetectionModel(nn.Module):

    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = DarkNet()
        self.fpn = FPN()
        self.h1 = Head(605, 5, 512)
        self.h2 = Head(605, 5, 512)
        self.h3 = Head(605, 5, 1024)

    def forward(self, x):
        x1, x2, x = self.backbone(x)
        #print(x1.shape, x2.shape, x.shape)
        x1, x2 = self.fpn(x1, x2, x)
        x_1, x_2 = self.h3(x)
        x2_1, x2_2 = self.h2(x2)
        x1_1, x1_2 = self.h1(x1)
        
        x_1 = x_1.permute(0, 2, 3, 1)
        x_2 = x_2.permute(0, 2, 3, 1)

        x2_1 = x2_1.permute(0, 2, 3, 1)
        x2_2 = x2_2.permute(0, 2, 3, 1)

        x1_1 = x1_1.permute(0, 2, 3, 1)
        x1_2 = x1_2.permute(0, 2, 3, 1)

        return x_1, x_2, x2_1, x2_2, x1_1, x1_2



class FPNALT(nn.Module):

    def __init__(self):
        super(FPNALT, self).__init__()
        self.up1 = UpSample()
        self.up2 = UpSample()
        self.up3 = UpSample()
        self.pool1 = SPPF(1024+512, 512)
        self.pool2 = SPPF(1024+256+512, 512)

        self.conv1 = ConvOne(1024, 1024, 1, 1) 
        self.conv2 = ConvOne(512, 512, 1, 1)
        self.conv3 = ConvOne(256, 256, 1, 1)
    
    def forward(self, x1, x2, x):
        ## x 16, 16 1024
        ## x2 32, 32 512
        ## x1 64 64 512
        x = self.conv1(x)
        x = self.up1(x)
        x2 = self.conv2(x2)
        x2 = torch.cat([x, x2], 1)
        x_ = self.pool1(x2)
        x2 = self.up2(x_)
        x1 = self.conv3(x1)
        x = self.up3(x)
        x1 = torch.cat([x1, x2, x], 1)
        x1 = self.pool2(x1)

        ## x1 64 512
        ## x_ 32 512



        ## conv layers 
        ## concat and pool
        return x1, x_


class HeadALT(nn.Module):
    ### use inverted residual block here
    ## output gives 
    def __init__(self, num_class=100, box=6, c=512):
        super(HeadALT, self).__init__()
        c2 = c // 2
        self.c1 = InvertedResidualConv(c, num_class+box, c2, 1, 1, 1)
        #self.c2 = InvertedResidualConv(c, num_class+box, c2, 1, 1, 1)
        

    def forward(self, x):
        x = self.c1(x)
        return x

class ObjectDetectionModelNoSeparation(nn.Module):

    def __init__(self):
        super(ObjectDetectionModelNoSeparation, self).__init__()
        self.backbone = DarkNet()
        self.fpn = FPNALT()
        self.h1 = HeadALT(605, 5, 512)
        self.h2 = HeadALT(605, 5, 512)
        self.h3 = HeadALT(605, 5, 1024)

    def forward(self, x):
        x1, x2, x = self.backbone(x)
        #print(x1.shape, x2.shape, x.shape)
        x1, x2 = self.fpn(x1, x2, x)
        x_1 = self.h3(x)
        x2_1 = self.h2(x2)
        x1_1 = self.h1(x1)
        
        x = x_1.permute(0, 2, 3, 1)
        x_1 = x[...,:605]
        x_2 = x[...,605:]

        x2 = x2_1.permute(0, 2, 3, 1)
        x2_1 = x2[...,:605]
        x2_2 = x2[...,605:]

        x1 = x1_1.permute(0, 2, 3, 1)
        x1_1 = x1[...,:605]
        x1_2 = x1[...,605:]

        #print(x_1.shape, x_2.shape, x2_1.shape, x2_2.shape, x1_1.shape, x1_2.shape)

        return x_1, x_2, x2_1, x2_2, x1_1, x1_2
