import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init_weights import init_weights
from functools import partial
from utils.attention import  CBAMs
nonlinearity = partial(F.relu, inplace=True)


class deconv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(deconv, self).__init__()
        self.depthwise = nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=1, padding=1, groups=inchannel,bias=False)
        self.pointwise = nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class Block(nn.Module ):
    def __init__(self,inchannel,outchannel,n):
        super(Block, self).__init__()
        self.n=n
        self.downsampple = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        for i in range(1,n+1):
            conv=nn.Sequential (
                nn.Conv2d (inchannel,outchannel,kernel_size=3,stride= 1,padding= 1,bias= False  ),
                nn.BatchNorm2d(outchannel ),
                nn.ReLU ()
            )
            setattr(self ,f'conv{i}d',conv)
            inchannel=outchannel



    def forward(self,x):
        residual=self.downsampple (x)
        for i in range(1,self.n+1):
            conv=getattr(self,f'conv{i}d' )
            x=conv(x)
        x=residual+x
        return x
class BasicBlock(nn.Module):
    def __init__(self,inchannel,channel,stride=1,dowmsample=None ):
        super(BasicBlock, self).__init__()

        self.conv1=deconv(inchannel ,channel)
        self.bn=nn.BatchNorm2d (channel)
        self.relu=nn.ReLU ()

        self.conv2=deconv (channel,channel)

        self.downsample=dowmsample

    def forward(self,x):

        residual=x

        x=self.conv1(x)
        x=self.bn(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn(x)

        if self.downsample is not None :
            residual=self.downsample(residual)

        x=x+residual

        return self.relu(x)

class Conv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.n = n
        self.ks = ks  # kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size, momentum=0.9),
                                     nn.ReLU(), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x
    





class atten_Up(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, ):
        super(atten_Up, self).__init__()
        self.conv = Conv(out_size * 2, out_size, False)

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.attention = CBAMs(in_size, reduction=16)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('Conv') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, input):

        _,_,h,w=input .shape
        outputs0 = self.up(inputs0)

        _,_,h1,w1=outputs0 .shape

        diffX=abs(h-h1)
        diffY=abs(w-w1)
        outputs0=F.pad(outputs0 ,[diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        outputs0 = torch.cat([outputs0, input], 1)
        return self.conv(self.attention(outputs0))
































