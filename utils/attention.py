import torch
import torch.nn as nn
import cv2
import numpy as np


#SENET
class SElayer(nn.Module ):
    def __init__(self,in_channel,reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d (1)
        self.se=nn.Sequential (
            nn.Linear (in_channel ,in_channel //reduction ,bias= False ),
            nn.ReLU (inplace= True ),
            nn.Linear (in_channel //reduction ,in_channel ,bias= False ),
            nn.Sigmoid ()
        )
    def forward(self,x):
        b,c,h,w=x.shape
        y=self .avg_pool (x).view(b,c)
        y=self .se (y).view(b,c,1,1)
        return x*y.expand_as(x)


#non-local
class Non_Local(nn.Module ):
    def __init__(self,in_channel):
        super(Non_Local, self).__init__()
        self.inter_channel=in_channel //2

        self.conv_key=nn.Conv2d (in_channel ,self .inter_channel ,kernel_size= 1,stride= 1,padding= 0,bias= False )
        self.conv_query=nn.Conv2d (in_channel ,self.inter_channel ,kernel_size= 1,stride= 1,padding= 0,bias= False )
        self.conv_value=nn.Conv2d (in_channel ,self .inter_channel ,kernel_size= 1,stride= 1,padding= 0,bias= False )

        self.Softmax=nn.Softmax(dim=1)

        self.conv_mask=nn.Conv2d(self .inter_channel ,in_channel ,kernel_size= 1,stride= 1,padding= 0,bias= False )
    def forward(self,x):
        b,c,h,w=x.shape

        key=self .conv_key (x).view(b,c,-1)
        query=self.conv_query (x).view(b,c,-1).permute(0,2,1).contiguous()
        value=self .conv_value (x).view(b,c,-1).permute(0,2,1).contiguous()

        atten=torch.matmul(query ,key )
        atten =self.Softmax (atten )

        mask=torch.matmul(atten ,value ).permute(0,2,1).contiguous().view(b,self.inter_channel ,h,w)
        mask=self.conv_mask (mask)

        return x+mask


#CBAM
class ChannelAttention(nn.Module ):
    def __init__(self,in_channel,reduction):
        super(ChannelAttention, self).__init__()

        self.max_pool=nn.AdaptiveMaxPool2d (1)
        self.avg_pool=nn.AdaptiveAvgPool2d (1)

        self.MLP=nn.Sequential (
            nn.Conv2d (in_channel ,in_channel //reduction ,1,bias= False ),
            nn.ReLU (),
            nn.Conv2d (in_channel //reduction ,in_channel ,1,bias=False )
        )
        self.sig=nn.Sigmoid ()

    def forward(self,x):

        avg_pool=self .MLP (self .avg_pool (x))
        max_pool=self.MLP (self.max_pool (x))
        return self .sig(max_pool +avg_pool )
class SpatialAttention(nn.Module ):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv=nn.Conv2d (2,1,kernel_size= 7,stride= 1,padding= 3)
        self.sig=nn.Sigmoid ()


    def forward(self,x):
        max_pool,_=torch.max(x,dim= 1,keepdim= True )
        avg_pool=torch.mean(x,dim= 1,keepdim= True )
        y=torch.cat([max_pool ,avg_pool ],dim= 1)
        out=self .conv (y)
        return self.sig (out)

class CBAM(nn.Module ):
    def __init__(self,in_channel,reduction):
        super(CBAM, self).__init__()
        self.Channel=ChannelAttention (in_channel ,reduction )
        self.Spatial=SpatialAttention ()

    def forward(self,x):
        channel=self .Channel (x)
        out=channel *x
        spatial=self .Spatial (out)
        out*=spatial
        return out


class RES_CBAM_block(nn.Module):

    def __init__(self,in_channel,channel,stride=1,downsample=False ,expansion=4):
        super(RES_CBAM_block, self).__init__()
        self.conv1=nn.Conv2d (in_channel ,channel ,kernel_size= 1,stride =1,padding= 1)
        self.bn1=nn.BatchNorm2d (channel )
        self.relu=nn.ReLU (inplace= True )

        self.conv2=nn.Conv2d (channel,channel,kernel_size= 3,stride=stride,padding= 0)
        self .bn2=nn.BatchNorm2d (channel )

        self .conv3=nn.Conv2d (channel ,channel *expansion ,kernel_size= 1,stride= 1,padding= 0)
        self.bn3=nn.BatchNorm2d (channel *expansion )
        self.CBAM=CBAM (channel *expansion ,16)

        if downsample :
            self.shortcut=nn.Conv2d (in_channel ,channel*expansion ,kernel_size= 1,stride =stride ,bias= False )
        else:
            self.shortcut=nn.Conv2d (in_channel ,channel *expansion ,kernel_size= 1,stride =1,bias= False )


    def forward(self,x):

        identity=self .shortcut (x)
        x=self .relu (self .bn1(self .conv1 (x)))
        x=self .relu(self .bn2(self .conv2 (x)))
        x=self .bn3(self .conv3 (x))
        x=self .CBAM (x)
        x+=identity

        return self.relu (x)


#Coordinate Attention
class h_swish(nn.Module):
    def __init__(self,inplace=True ):
        super(h_swish, self).__init__()
        self.relu=nn.ReLU6 (inplace= inplace )

    def forward(self,x):
        sigmoid=self.relu(x+3) / 6
        x=x*sigmoid
        return x
class CoorAttention(nn.Module ):
    def __init__(self,in_channels,out_channels,reduction=32):
        super(CoorAttention, self).__init__()
        self.pool_x=nn.AdaptiveAvgPool2d ((None ,1))
        self.pool_y=nn.AdaptiveAvgPool2d ((1,None ))

        middle_channel=max(8,in_channels //reduction )

        self.conv1=nn.Conv2d (in_channels ,middle_channel ,kernel_size= 1,stride= 1,padding= 0)
        self.bn1=nn.BatchNorm2d (middle_channel )

        self.non_linear=h_swish ()

        self.convx=nn.Conv2d (middle_channel ,out_channels ,kernel_size= 1,stride= 1,padding= 0)
        self.convy=nn.Conv2d (middle_channel ,out_channels ,kernel_size= 1,stride= 1,padding= 0)
        self.sigmoid=nn.Sigmoid ()

    def forward(self,x):
        identity=x

        b,c,h,w=x.shape

        pool_h=self .pool_x (x)
        pool_w=self.pool_y (x)
        pool_w =pool_w.permute(0,1,3,2)

        pool=torch.cat([pool_h ,pool_w],dim=2)

        pool=self.non_linear (self.bn1 (self.conv1 (pool)))

        x_h, x_w = torch.split(pool, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        attention_h = self.sigmoid(self.convx(x_h))
        attention_w = self.sigmoid(self.convy(x_w))
        return identity * attention_h * attention_w


#CBAM改

class Channel(nn.Module ):
    def __init__(self,inchannel,reduction=16):
        super(Channel, self).__init__()
        radio=reduction //2


        self.gap=nn.AdaptiveAvgPool2d(1)
        self.gmp=nn.AdaptiveMaxPool2d (1)

        self.mlp=nn.Sequential (
            nn.Conv2d(inchannel ,inchannel //radio ,1,1,0,bias= False ),
            nn.ReLU (),
            nn.Conv2d (inchannel //radio,inchannel//reduction ,1,1,0,bias= False ),
            nn.ReLU (),
            nn.Conv2d (inchannel //reduction ,inchannel//radio ,1,1,0,bias= False ),
            nn.ReLU (),
            nn.Conv2d (inchannel//radio,inchannel ,1,1,0,bias= False )
        )
        self.sig=nn.Sigmoid ()

    def forward(self,x):

        gap=self.gap (x)
        gmp=self.gmp(x)

        gap=self.mlp(gap)
        gmp=self.mlp(gmp)


        return self.sig(gap+gmp)


class Spatial(nn.Module ):
    def __init__(self):
        super(Spatial, self).__init__()

        self.conv=nn.Sequential (

            nn.Conv2d (in_channels= 2,out_channels= 2,kernel_size= 3,stride= 1,padding= 1,bias= False ),
            nn.ReLU (),
            nn.Conv2d (in_channels= 2,out_channels= 2,kernel_size= 3,stride= 1,padding= 1,bias= False ),
            nn.ReLU (),
            nn.Conv2d (in_channels= 2,out_channels= 1,kernel_size= 3,stride= 1,padding= 1,bias= False )
        )

        self.sig=nn.Sigmoid ()

    def forward(self,x):

        max_pool,_=torch.max (x,dim=1,keepdim= True)
        avg_pool  =torch.mean(x,dim=1,keepdim= True)

        cc=torch.cat([max_pool ,avg_pool ],dim= 1)

        cc=self.conv (cc)

        cc=self.sig(cc)

        return cc


class CBAMs(nn.Module ):
    def __init__(self,inchannel,reduction):
        super(CBAMs, self).__init__()

        self.channel=Channel (inchannel ,reduction )
        self.spatial=Spatial ()

    def forward(self,x):

        ca=self.channel (x)
        out=ca*x
        sa=self.spatial (out)
        out=out*sa

        return out





