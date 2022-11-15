import torch.nn as nn
import torch.nn.functional as F
import torch as t
import math
from pysot.models.utile.tran import Transformer

class hiftmodule(nn.Module):
    
    def __init__(self,cfg):
        super(hiftmodule, self).__init__()
        
        
        channel=256
        self.conv1=nn.Sequential(
                nn.Conv2d(256, 256,  kernel_size=3, stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                )
        self.conv2 = nn.Sequential(
               nn.ConvTranspose2d(256*2, 256,  kernel_size=1, stride=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
               ) 
        self.conv3 = nn.Sequential(
               nn.ConvTranspose2d(256, 256,  kernel_size=2, stride=2),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
               nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
               
              
             
                ) 
        
        self.convloc = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=2, stride=2),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),                
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 4,  kernel_size=3, stride=1,padding=1),
                )
        
        self.convcls = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=2, stride=2),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.row_embed1 = nn.Embedding(50, 256//2)
        self.col_embed1 = nn.Embedding(50, 256//2)
        self.row_embed2 = nn.Embedding(50, 256//2)
        self.col_embed2 = nn.Embedding(50, 256//2)
        self.reset_parameters()
        #self.scr = SCRattention(channel,channel)
        self.trans = Transformer(256, 4,1,1)
        
        self.cls1=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(channel, 1,  kernel_size=3, stride=1,padding=1)
        for modules in [self.conv1,self.conv2,self.convloc, self.convcls,self.conv3,
                        self.cls1, self.cls2]:
            for l in modules.modules():
               if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
        
        
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed1.weight)
        nn.init.uniform_(self.col_embed1.weight)
        nn.init.uniform_(self.row_embed2.weight)
        nn.init.uniform_(self.col_embed2.weight)
        
    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def forward(self,zf,xf):
        
        # resx=self.xcorr_depthwise(x, z)
        # resd=self.xcorr_depthwise(xf, zf)
        # resd=self.conv1(resd)
        # res=self.conv2(t.cat((resx,resd),1))
        h1, w1 = 7, 7
        i1 = t.arange(w1).cuda()
        j1 = t.arange(h1).cuda()
        x_emb1 = self.col_embed1(i1)
        y_emb1 = self.row_embed1(j1)
        # print("x_emb1.shape:",x_emb1.shape)
        # print("y_emb1.shape:", y_emb1.shape)
        pos1 = t.cat([
            x_emb1.unsqueeze(0).repeat(h1, 1, 1),
            y_emb1.unsqueeze(1).repeat(1, w1, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(zf.shape[0], 1, 1, 1)
        #print("pos1.shape:", pos1.shape)
        h2, w2 = 26, 26
        i2 = t.arange(w2).cuda()
        j2 = t.arange(h2).cuda()
        x_emb2 = self.col_embed2(i2)
        y_emb2 = self.row_embed2(j2)
        # print("x_emb2.shape:",x_emb2.shape)
        # print("y_emb2.shape:", y_emb2.shape)
        pos2 = t.cat([
            x_emb2.unsqueeze(0).repeat(h2, 1, 1),
            y_emb2.unsqueeze(1).repeat(1, w2, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(xf.shape[0], 1, 1, 1)
        #print("pos2.shape:", pos2.shape)
        zzz=pos1 + zf
        #print("zf+pos1:",zzz.shape)

        b1, c1, w1, h1=zf.size()
        b2, c2, w2, h2=xf.size()

        xxx=pos2 + xf
        #print("xf+pos2:",xxx.shape)
        #print("t1:",(pos1+zf).view(b1,c1,-1).permute(2, 0, 1).shape)
        #print("t2:", (pos2 + xf).view(b2, c2, -1).permute(2, 0, 1).shape)
        res2=self.trans((pos1+zf).view(b1,c1,-1).permute(2, 0, 1), \
                        (pos2+xf).view(b2,c2,-1).permute(2, 0, 1))
        #print("res2.shape:",res2.shape)
        res2=res2.permute(1,2,0).view(b2,256,w2,h2)
        #print("tf_out.shape:",res2.shape)
        return res2

############################################
###########################################


##########################################
###########################################

class scr_attention(nn.Module):

    def __init__(self, cfg):
        super(scr_attention, self).__init__()

        channel = 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256 * 2, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),

        )

        self.convloc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 4, kernel_size=3, stride=1, padding=1),
        )

        self.convcls = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=2, stride=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.row_embed1 = nn.Embedding(50, 256 // 2)
        self.col_embed1 = nn.Embedding(50, 256 // 2)
        self.row_embed2 = nn.Embedding(50, 256 // 2)
        self.col_embed2 = nn.Embedding(50, 256 // 2)
        self.reset_parameters()
        self.scr = scr_module(channel, channel)

        self.cls1 = nn.Conv2d(channel, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)
        for modules in [self.conv1, self.conv2, self.convloc, self.convcls, self.conv3,
                        self.cls1, self.cls2]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed1.weight)
        nn.init.uniform_(self.col_embed1.weight)
        nn.init.uniform_(self.row_embed2.weight)
        nn.init.uniform_(self.col_embed2.weight)

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def forward(self, y):
        y = self.scr(y)

        # resx = self.xcorr_depthwise(x, z)
        # resd = self.xcorr_depthwise(xf, zf)
        # resd = self.conv1(resd)
        # res = self.conv2(t.cat((resx, resd), 1))
        # h1, w1 = 11, 11
        # i1 = t.arange(w1).cuda()
        # j1 = t.arange(h1).cuda()
        # x_emb1 = self.col_embed1(i1)
        # y_emb1 = self.row_embed1(j1)
        #
        # pos1 = t.cat([
        #     x_emb1.unsqueeze(0).repeat(h1, 1, 1),
        #     y_emb1.unsqueeze(1).repeat(1, w1, 1),
        # ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(res.shape[0], 1, 1, 1)
        #
        # h2, w2 = 22, 22
        # i2 = t.arange(w2).cuda()
        # j2 = t.arange(h2).cuda()
        # x_emb2 = self.col_embed2(i2)
        # y_emb2 = self.row_embed2(j2)
        #
        # pos2 = t.cat([
        #     x_emb2.unsqueeze(0).repeat(h2, 1, 1),
        #     y_emb2.unsqueeze(1).repeat(1, w2, 1),
        # ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(res.shape[0], 1, 1, 1)
        #
        # b, c, w, h = res.size()
        # res1 = self.conv3(res)
        # res2 = self.trans((pos1 + res).view(b, 256, -1).permute(2, 0, 1), \
        #                   (pos2 + res1).view(b, 256, -1).permute(2, 0, 1))
        #
        # res2 = res2.permute(1, 2, 0).view(b, 256, 22, 22)
        # loc = self.convloc(res2)
        # acls = self.convcls(res2)
        #
        # cls1 = self.cls1(acls)
        # cls2 = self.cls2(acls)
        # print("scr_attention_out.shape:",y.shape)
        loc = self.convloc(y)
        acls = self.convcls(y)

        cls1 = self.cls1(acls)
        cls2 = self.cls2(acls)
        return loc, cls1, cls2


class scr_module(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(scr_module, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')


        self.conv_q_right.inited = True
        self.conv_v_right.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = t.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out
    def forward(self, x):

        out = self.spatial_pool(x)


        # s, b, c = y.size()
        #
        # w = int(pow(s, 0.5))
        # y = y.permute(1, 2, 0).view(b, c, w, w)
        # ww = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y)))))
        # x = x.permute(1, 2, 0).view(b, c, 22, 22)
        # m = x + self.gamma * ww * x
        # m = m.view(b, c, -1).permute(2, 0, 1)
        return out

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


