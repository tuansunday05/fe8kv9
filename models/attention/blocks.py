# Shuffle 
# CBAM
# -- GAM ECA SE SK LSK
from models.common import *

class RepNCBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(CBAMBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
class RepNSA(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, groups=8, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(SABottleneck(c_, c_, 1, shortcut, groups=groups) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
# ----------------------- Attention Mechanism ---------------------------

## CBAM ATTENTION
class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.act = nn.SiLU()

        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.act(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.act(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #2*h*w
        x = self.conv(x)
        #1*h*w
        return self.sigmoid(x)
    
class CBAMBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 c1,
                 c2,
                 shortcut=True,
                 g=1,
                 e=0.5,
                 ratio=16,
                 kernel_size=3):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        #self.cbam=CBAM(c1,c2,ratio,kernel_size)

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        out = self.channel_attention(x1) * x1
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return x + out if self.add else out

class CBAMC4(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super(CBAMC4, self).__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
        self.channel_attention = ChannelAttention(c2)
        self.spatial_attention = SpatialAttention(kernel_size=3)  # Specify kernel_size here

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        y = torch.cat(y, 1)
        
        # Apply channel attention
        y = y * self.channel_attention(y)
        
        # Apply spatial attention
        y = y * self.spatial_attention(y)
        
        return self.cv4(y)

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        y = torch.cat(y, 1)
        
        # Apply channel attention
        y = y * self.channel_attention(y)
        
        # Apply spatial attention
        y = y * self.spatial_attention(y)
        
        return self.cv4(y)

class RepNCBAMELAN4(RepNCSPELAN4):
    # C3 module with CBAMBottleneck()
    def __init__(self, c1, c2, c3, c4, c5=1): 
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(RepNCBAM(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCBAM(c4, c4, c5), Conv(c4, c4, 3, 1))

        # c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepCBAM(c_, c_, shortcut) for _ in range(n)))

## GAM ATTETION
class GAMAttention(nn.Module):
       #https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True,rate=4):
        super(GAMAttention, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1//rate, kernel_size=7, padding=3,groups=rate)if group else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3), 
            nn.BatchNorm2d(int(c1 /rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1//rate, c2, kernel_size=7, padding=3,groups=rate) if group else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3), 
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
 
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att=channel_shuffle(x_spatial_att,4) #last shuffle 
        out = x * x_spatial_att
        return out  

def channel_shuffle(x, groups=2):
        B, C, H, W = x.size()
        out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
        out=out.view(B, C, H, W) 
        return out


## SK ATTENTION

# class SKAttention(nn.Module):

#     def __init__(self, channel=512,out_channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
#         super().__init__()
#         self.d=max(L,channel//reduction)
#         self.convs=nn.ModuleList([])
#         for k in kernels:
#             self.convs.append(
#                 nn.Sequential(OrderedDict([
#                     ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
#                     ('bn',nn.BatchNorm2d(channel)),
#                     ('relu',nn.ReLU())
#                 ]))
#             )
#         self.fc=nn.Linear(channel,self.d)
#         self.fcs=nn.ModuleList([])
#         for i in range(len(kernels)):
#             self.fcs.append(nn.Linear(self.d,channel))
#         self.softmax=nn.Softmax(dim=0)

#     def forward(self, x):
#         bs, c, _, _ = x.size()
#         conv_outs=[]
#         ### split
#         for conv in self.convs:
#             conv_outs.append(conv(x))
#         feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

#         ### fuse
#         U=sum(conv_outs) #bs,c,h,w

#         ### reduction channel
#         S=U.mean(-1).mean(-1) #bs,c
#         Z=self.fc(S) #bs,d

#         ### calculate attention weight
#         weights=[]
#         for fc in self.fcs:
#             weight=fc(Z)
#             weights.append(weight.view(bs,c,1,1)) #bs,channel
#         attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
#         attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

#         ### fuse
#         V=(attention_weughts*feats).sum(0)
#         return V
    
## SHUFFLE ATTENTION
from torch.nn.parameter import Parameter
from torch.nn import init

class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(self.channel // (2 * self.G), self.channel // (2 * self.G))
        self.cweight = Parameter(torch.zeros(1, self.channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, self.channel // (2 * self.G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, self.channel // (2 * self.G), 1, 1))
        self.sbias = Parameter(torch.ones(1, self.channel // (2 * self.G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.reshape(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().reshape(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

    
class SABottleneck(nn.Module):
    def __init__(self, c1, c2, stride=1, shortcut = True, reduction=16, groups=8):
        super().__init__()
        c__ = c2 // 2
        self.conv1 = Conv(c1, c__, 1, 1, g= groups)
        self.conv2 = Conv(c__, c__, 3, stride, 1, g= groups)
        self.conv3 = Conv(c__, c2, 1, 1, g= groups, act=False)
        self.act = nn.SiLU()

        self.shuffle_attention = ShuffleAttention(channel=c2, G=groups)
        self.downsample = None
        self.shortcut = shortcut
        if stride != 1 or c1 != c2:
            self.downsample = nn.Sequential(
                Conv(c1, c2, k=1, stride=stride, g=groups, act=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += self.shuffle_attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.shortcut:
            out += residual
        out = self.act(out)

        return out

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)

# class sa_layer(nn.Module):
#     """Constructs a Channel Spatial Group module.

#     Args:
#         k_size: Adaptive selection of kernel size
#     """

#     def __init__(self, channel, groups=64):
#         super(sa_layer, self).__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
#         self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

#         self.sigmoid = nn.Sigmoid()
#         self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, h, w = x.shape

#         x = x.reshape(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)

#         # flatten
#         x = x.reshape(b, -1, h, w)

#         return x

#     def forward(self, x):
#         b, c, h, w = x.shape

#         x = x.reshape(b * self.groups, -1, h, w)
#         x_0, x_1 = x.chunk(2, dim=1)

#         # channel attention
#         xn = self.avg_pool(x_0)
#         xn = self.cweight * xn + self.cbias
#         xn = x_0 * self.sigmoid(xn)

#         # spatial attention
#         xs = self.gn(x_1)
#         xs = self.sweight * xs + self.sbias
#         xs = x_1 * self.sigmoid(xs)

#         # concatenate along channel axis
#         out = torch.cat([xn, xs], dim=1)
#         out = out.reshape(b, -1, h, w)

#         out = self.channel_shuffle(out, 2)
#         return out


# class SABottleneck(nn.Module):
#     expansion = 4
#     __constants__ = ['downsample']

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=8,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(SABottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 256.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes)
#         self.bn3 = norm_layer(planes)
#         self.sa = sa_layer(planes, 8)
#         self.relu = nn.ReLU()
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.sa(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

class RepNSAELAN4(RepNCSPELAN4):
    # C3 module with CBAMBottleneck()
    def __init__(self, c1, c2, c3, c4, c5=1): 
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(RepNSA(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNSA(c4, c4, c5), Conv(c4, c4, 3, 1))

## ECA
class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return out * x
    
class ECABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 c1,
                 c2,
                 shortcut=True,
                 g=1,
                 e=0.5,
                 ratio=16,
                 k_size=3):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.eca=ECA(c1,c2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        # out=self.eca(x1)*x1
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x1 * y.expand_as(x1)

        return x + out if self.add else out
    
## LSK Attention
class LSKblock(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.conv0 = nn.Conv2d(c1, c1, 5, padding=2, groups=c1)
        self.conv_spatial = nn.Conv2d(c1, c1, 7, stride=1, padding=9, groups=c1, dilation=3)
        self.conv1 = nn.Conv2d(c1, c1//2, 1)
        self.conv2 = nn.Conv2d(c1, c1//2, 1)
        # self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(c1//2, c1, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn



class LSKAttention(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.proj_1 = nn.Conv2d(c1, c1, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(c1)
        self.proj_2 = nn.Conv2d(c1, c1, 1)
        self.proj_3 = nn.Conv2d(c1, c2, 1)


    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        x = self.proj_3(x)
        return x



## SE Attention
class SEBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.se=SE(c1,c2,ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        b, c, _, _ = x.size()
        y = self.avgpool(x1).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        out = x1 * y.expand_as(x1)

        # out=self.se(x1)*x1
        return x + out if self.add else out

## SOCA Attention
from torch.autograd import Function

class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) - 
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) - 
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) - 
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

class SOCA(nn.Module):
    # Second-order Channel Attention
    def __init__(self, c1, c2, reduction=8):
        super(SOCA, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_du = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1, padding=0, bias=True),
            nn.SiLU(),  # SiLU activation
            nn.Conv2d(c1 // reduction, c1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        cov_mat = CovpoolLayer(x_sub)  # Global Covariance pooling layer
        cov_mat_sqrt = SqrtmLayer(cov_mat, 5)  # Matrix square root layer (including pre-norm, Newton-Schulz iter. and post-com. with 5 iterations)
        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(batch_size, C, 1, 1)
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov * x
