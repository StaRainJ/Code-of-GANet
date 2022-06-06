import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import math
from loss import batch_episym
from sinkhorn_norm import SparseAttention, SparseCutAttention

class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, pre=True):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1), stride=(1, stride)#, padding=(0, 1)
            ),
            nn.BatchNorm2d(outchannel)
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1), stride=(1, stride)#, padding=(0, 1)
            ),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, (1, 1), stride=1#, padding=(0, 1)
            ),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel)
        )
    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)       
        out = out + x1
        return F.relu(out)

class Group_Attention(nn.Module):
    def __init__(self, inchannel, outchannel, weight_out= False):
        super(Group_Attention, self).__init__()
        self.group = 6
        self.cut_length = 3
        self.group_norm = nn.GroupNorm(self.group, inchannel)
        self.v_activation = F.elu
        self.linear = nn.Conv2d(inchannel, inchannel, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)
        self.weight_out = weight_out
        self.resnet = ResNet_Block(inchannel, outchannel, stride=1, pre=True)
        self.sattn = SparseCutAttention(blocks=10, temperature= (inchannel ** .5), sinkhorn_iter=8, inchannel=inchannel, cut_length=self.cut_length)
    
    def forward(self, x):
    
        batch_size, c, num_pts = x.shape[0], x.shape[1], x.shape[2]
        
        feat = self.linear(x)                            # B * C * H * 1
        feat_groups = feat.chunk(self.group, dim=-1)     # B * C_group * H * 1
        
        #multi-head attention
        attn_groups = []
        for group in feat_groups:
            q = group
            k = group
            v = self.v_activation(group)                 # B * C_group * H * 1
            
            '''
            attn_logits = torch.matmul(q, k.transpose(2,3)) / temperature     # B * Groups * H * H
            attn = F.softmax(attn_logits, dim=-1))
            attn_groups.append(torch.matmul(attn, v))          # B * Groups * H * C_group
            '''
            #sparse attention
            attn = self.sattn(q, k, v)
            attn_groups.append(attn)
        
        feat_attn = torch.cat(attn_groups, dim=-1)       # B * 1 * H * C
        feat_attn = feat_attn.transpose(1,3)             # B * C * H * 1
        feat_attn = self.group_norm(feat_attn + x)
        
        mean = torch.sum(attn, dim = -2)
        mean = torch.div(mean, num_pts)
        
        x_out = self.resnet(feat_attn)
        #print('x_out.shape', x_out.shape)
        
        if self.weight_out == True:
            return x_out, mean
        else:
            return x_out
        

class ATEM(nn.Module):
    def __init__(self, net_channels=256, input_channel=4, ):
        super(ATEM, self).__init__()
        #self.conv = nn.Conv2d(1, net_channels, (1, 2), stride=(1, 2),bias=True)
        #self.gconv = nn.Conv2d(1, 1, (1, input_channel), stride=(1, input_channel),bias=True)
        self.norm = nn.InstanceNorm2d(net_channels)  
        #self.em = EM(net_channels, 64, 3)      # 1 * c *iteration
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, net_channels, (1, input_channel), stride=(1, input_channel),bias=True),
            nn.BatchNorm2d(net_channels),
            nn.ReLU(),
        )
                 
        self.conv2 = nn.Sequential(
            nn.Conv2d(net_channels, net_channels, (1, 1)),
            nn.InstanceNorm2d(net_channels),
            nn.BatchNorm2d(net_channels),
            nn.ReLU(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(net_channels, 1, (1, 1)),
        ) 
        self.layer = self.make_layers(ResNet_Block, Group_Attention, net_channels)
        
    def make_layers(self, ResNet_Block, Group_Attention, net_channels):
        layers = []
        layers.append(Group_Attention(net_channels, net_channels, weight_out= False))
        layers.append(Group_Attention(net_channels, net_channels, weight_out= False))
        layers.append(Group_Attention(net_channels, net_channels, weight_out= False))
        layers.append(Group_Attention(net_channels, net_channels, weight_out= True))
        return nn.Sequential(*layers)
        
    
    def forward(self, x):
    
        out = self.conv1(x)
        batch_size, num_pts = x.shape[0], x.shape[2]
        
        #resnet block and attention
        out, mean = self.layer(out)
        
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        out = out
        w = torch.tanh(out)
        w = F.relu(w)
        
        mean = torch.tanh(mean)
        mean = F.relu(mean)
        
        e_hat = weighted_8points(x[:,:,:,:4], w)

        x1, x2 = x[:,0,:,:2], x[:,0,:,2:4]
        #e_hat_norm = e_hat
        #residual = batch_episym(x1, x2, e_hat).reshape(batch_size, 1, num_pts, 1)
        
        return out, e_hat#, residual, mean


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv
    
    
    
def weighted_8points(x_in, weights):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    #weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat