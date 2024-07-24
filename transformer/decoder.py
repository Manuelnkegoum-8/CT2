import torch
import  torch.nn as  nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import random,math
import torch.nn.functional as F
from .encoder import feedforward


class ColorAttention(nn.Module):
    def __init__(self,embed_dim=512,ff_hidden_dim=1024,dropout_rate = 0.1,patch_size=16,num_heads=8):
        super().__init__()
        self.patch_mask = nn.Sequential(
            Rearrange("b (h p1) (w p2) c -> b (h w)  (p1 p2) c", p1 = patch_size, p2 = patch_size),
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tau = torch.sqrt(torch.tensor(self.head_dim))
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim,bias = False)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.patch_size = patch_size

        
    def _masked_softmax(self, attention_scores, attention_mask):
        if attention_mask is not None:
            #attention_mask B*196+313*196+313
            expand_mask = attention_mask.unsqueeze(1).repeat(1,self.num_heads,1,1)
            attention_scores = attention_scores.masked_fill(expand_mask == 0, -float('inf'))
        return F.softmax(attention_scores, dim=-1)
    

    def process_mask(self, bs,seq_len,mask):
        # mask B*h*w*313
        mask_ = self.patch_mask(mask) # B*n*p2*313
        mask_ = mask_.sum(dim=2) # union of p2 patches B*n*313
        n_cls = mask_.size(-1)
        n1 = seq_len+n_cls
        full_mask = torch.full((bs,n1,n1),1.).to(mask.device)
        full_mask[:,:seq_len,seq_len:] = mask_
        full_mask[:,seq_len:,:seq_len] = mask_.transpose(1,2)
        return full_mask
    
    def forward(self,x,colors,mask):
        bs,seq_len,_ = x.size()
        attn_mask = self.process_mask(bs,seq_len,mask)
        inputs = torch.cat((x,colors),dim=1)
        qkv = self.qkv_proj(inputs).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        attention_scores = torch.matmul(q, k.transpose(-2,-1)) / self.tau
        attention_scores = self._masked_softmax(attention_scores,attn_mask)
        attention_output = torch.matmul(attention_scores, v).permute(0,2,1,3).contiguous().view(bs,-1,self.embed_dim)
        return self.o_proj(attention_output)


class ColorTransformerLayer(nn.Module):
    def __init__(self,embed_dim=512,ff_hidden_dim=1024,dropout_rate = 0.1,patch_size=16,num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.color_att = ColorAttention(embed_dim=embed_dim,
                                        ff_hidden_dim=ff_hidden_dim,
                                        dropout_rate =dropout_rate,
                                        patch_size=patch_size,
                                        num_heads=num_heads
                                        )
        self.ffd = feedforward(embed_dim=embed_dim,ff_hidden_dim=ff_hidden_dim)

    def forward(self, x,colors,masks):
        seq_len = x.size(1)
        inputs = torch.cat((x,colors),dim=1)
        out = self.norm(inputs)
        x_,cols_ = out[:,:seq_len], out[:,seq_len:]
        out = self.color_att(x_,cols_,masks) + inputs
        out = out + self.ffd(self.final_norm(out))
        return out[:,:seq_len], out[:,seq_len:]

class ColorTransformer(nn.Module):
    def __init__(self,embed_dim=512,ff_hidden_dim=1024,dropout_rate = 0.1,height=224,width=224,patch_size=16,num_heads=8,depth=2):
        super().__init__()
        self.block = nn.ModuleList()
        self.patch_tokens_in = Rearrange("b (h w) c -> b h w c", h = height//patch_size, w = width//patch_size)
        self.patch_tokens_out = Rearrange("b h w c -> b (h w) c", h = height//patch_size, w = width//patch_size)
        self.conv = nn.Conv2d(in_channels=embed_dim,out_channels=embed_dim,kernel_size=3,padding=1)
        self.linear = nn.Linear(embed_dim,embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        for _ in range(depth):
            layer = ColorTransformerLayer(embed_dim=embed_dim,
                                        ff_hidden_dim=ff_hidden_dim,
                                        dropout_rate =dropout_rate,
                                        patch_size=patch_size,
                                        num_heads=num_heads)
            self.block.append(layer)

    def forward(self, x,colors,masks):
        for layer in self.block:
            x,colors = layer(x,colors,masks)
        x = self.patch_tokens_in(x).permute(0,3,1,2)
        x = self.conv(x).permute(0,2,3,1)
        x = self.patch_tokens_out(x)
        colors = self.linear(colors)
        seq_len = x.size(1)
        out = torch.cat((x, colors), dim=1)
        out = self.norm(out)
        x,colors = out[:,:seq_len], out[:,seq_len:]
        x = self.patch_tokens_in(x)
        return x,colors


class UpsamplingBlock(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.up = nn.Sequential(
                nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=3,padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=out_dim,out_channels=out_dim,kernel_size=4,padding=1,stride=2,bias=True),
        )

    def forward(self, x):
        return self.up(x)

class Upsampling(nn.Module):
    def __init__(self,hidden_dims=[512,256,128,64,32]):
        super().__init__()
        self.block = nn.ModuleList()
        num_upsampling = len(hidden_dims)
        for i in range(num_upsampling-1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i+1]
            self.block.append(UpsamplingBlock(in_dim,out_dim))

    def forward(self, x):
        # B*h*w*512
        out = x.permute(0,3,1,2)
        for up in self.block:
            out = up(out)
        return out



class Decoder(nn.Module):
    def __init__(self,encoder_dim=512,embed_dim=512,ff_hidden_dim=1024,dropout_rate = 0.1,height=224,width=224,patch_size=16,num_heads=8,depth=2,num_upsampling=4):
        super().__init__()
        self.color_transformer = ColorTransformer(embed_dim=embed_dim,
                                                  ff_hidden_dim=ff_hidden_dim,
                                                  dropout_rate=dropout_rate,
                                                  height=height,
                                                  width=width,
                                                  patch_size=patch_size,
                                                  num_heads=num_heads,
                                                  depth=depth)
        self.norm = nn.LayerNorm(313)
        self.upsampler = Upsampling(hidden_dims= [embed_dim for _ in range(num_upsampling+1)])
        hidden_dims = [embed_dim//(2**i) for i in range(num_upsampling+1)]
        self.final_conv = nn.Conv2d(hidden_dims[-1],2,kernel_size=3,stride=1,padding=1)
        self.ab_upsampling = Upsampling(hidden_dims)

        self.proj = nn.Linear(encoder_dim,embed_dim)
        self.proj_patches = nn.Linear(embed_dim,embed_dim)
        self.proj_col = nn.Linear(embed_dim,embed_dim)
        self.apply(init_weights)
    def ColorQuery(self,x,colors,mask):
        # B*h*w*512 x
        # B*313*512 cols
        bs,h,w,dim = x.size()
        x = x.contiguous().view(bs,-1,dim)
        x = self.proj_patches(x)
        colors = self.proj_col(colors)

        norm_im = x/x.norm(dim=-1,keepdim=True)
        norm_cols = colors/colors.norm(dim=-1,keepdim=True)
        attention = self.norm(norm_im@(norm_cols.transpose(1,2)))
        attention = attention.masked_fill(mask==0,-float('inf'))
        return attention
    
    def forward(self,encoder_out,colors_tokens,mask): 
        encoder_out = self.proj(encoder_out)
        x,cols = self.color_transformer(encoder_out,colors_tokens,mask)
        pred_ab = F.tanh(self.final_conv(self.ab_upsampling(x)))
        x = self.upsampler(x).permute(0,2,3,1) #b*H*W*512
        bs,h,w,_ = x.size()
        pred_q = self.ColorQuery(x,cols,mask) #x*313
        pred_q = pred_q.contiguous().view(bs,h,w,-1).permute(0,3,1,2)
        return pred_q,pred_ab


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)