
import torch
import  torch.nn as  nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import random,math
from utils.quantization import *
from torch.autograd import Function
from utils.quantization import CIELAB

class GetClassWeights:
    def __init__(self, cielab, lambda_=0.5):
        prior = torch.from_numpy(cielab.gamut.prior)
        uniform = torch.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum().type_as(uniform)

        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)
    def __call__(self, ab_actual):
        weights = self.weights.to(ab_actual.device)
        return weights[ab_actual.argmax(dim=1, keepdim=True)]
    
class RebalanceLoss(Function):
    @staticmethod
    def forward(ctx, data_input, weights):
        ctx.save_for_backward(weights)

        return data_input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors

        # reweigh gradient pixelwise so that rare colors get a chance to
        # contribute
        grad_input = grad_output * weights

        # second return value is None since we are not interested in the
        # gradient with respect to the weights
        return grad_input, None
def encode(gt_ab,sigma,neighbours,q_ab,bins): # to define to get soft encoded labels Iqab
        """
        Parameters:
        - Iab: ntensor  (Bs,2,H,W) containing the continuous ab values.
        - quantized_ab: numpy array of shape (313, 2) containing the quantized ab pairs.
        
        Returns:
        - Iq: tensor (bs,313,H,W) containing the normalized soft labels.
        """
        
        sigma = sigma
        bs,_,H,W = gt_ab.size()
        Iab = gt_ab.permute(1,0,2,3).reshape(2,-1)
        q_ab = torch.from_numpy(q_ab).to(gt_ab.device).type(Iab.dtype)
        # Compute pairwise distances between Iab and quantized_ab
        distances = torch.cdist(q_ab,Iab.t())
        # Get the 5-nearest neighbors and their distances
        #nearest_distances, nearest_indices = torch.topk(distances, k=neighbours, dim=0, largest=False)
        nearest_indices = distances.argsort(dim=0)[:neighbours,:]
        nn_gauss = gt_ab.new_zeros(neighbours, bs*H*W)
        for i in range(neighbours):
            nn_gauss[i, :] = _gauss_eval(
                q_ab[nearest_indices[i, :], :].t(), Iab,sigma)

        nn_gauss /= nn_gauss.sum(dim=0, keepdim=True)

        # expand
        bins = bins

        q = gt_ab.new_zeros(bins, bs*H*W)

        q[nearest_indices, torch.arange(bs*H*W).repeat(neighbours, 1)] = nn_gauss
        # return: [bs,h,w,313]
        return q.reshape(bins,bs,H,W).permute(1, 0, 2,3)
    
def _gauss_eval(x, mu, sigma):
    norm = 1 / (2 * math.pi * sigma)
    return norm * torch.exp(-torch.sum((x - mu)**2, dim=0) / (2 * sigma**2))

class CondPositional(nn.Module):
    def __init__(self,height_patch=14,width_patch=14,dim=512,ks=3,color=None):
        super().__init__()
        self.patch_in = Rearrange("b (h w) c -> b h w  c", h = height_patch, w = width_patch)
        
        self.patch_out = nn.Sequential(
            Rearrange("b h w  c -> b (h w) c", h = height_patch, w = width_patch),
        )
        self.color = color
        self.conv = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=ks,padding=(ks-1)//2)

    def forward(self, inputs,h=None,w=None):
        if self.color is None :
            # inputs B,N,3
            #x bs,h,w,c
            out = self.patch_in(inputs).permute(0,3,1,2)
            out = self.conv(out).permute(0,2,3,1)
            out = self.patch_out(out)+inputs
        else:
            bs,_,c = inputs.size()
            cnn_feat = torch.zeros(bs,h,w,c).to(inputs.device)
            bin = 10
            torch_ab = torch.from_numpy(self.color).to(inputs.device)
            new_ab = torch.div(torch_ab + 110, bin, rounding_mode='floor')
            cnn_feat[:, new_ab[:, 0].long(), new_ab[:, 1].long(), :] = inputs
            out = cnn_feat.permute(0,3,1,2)
            out = self.conv(out).permute(0,2,3,1)
            pos = torch.zeros_like(inputs)
            pos[:, :, :] = out[:, new_ab[:, 0].long(), new_ab[:, 1].long(), :]
            out = pos+inputs
        return out
    

class CT2(nn.Module):
    def __init__(self,encoder,decoder,height=224,width=224,patch_size=16,dim=512,neighbours=5,num_quantized=313,sigma=5.0,q_ab=None,gamma_l1=1.,weight_l1=10.,device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sigma = sigma
        self.bins = num_quantized
        self.neighbours = neighbours
        self.q_ab = q_ab # to define ?? [313,2] shape 
        self.weight_l1 = weight_l1
        self.colors_tokens = nn.Parameter(torch.randn(1,num_quantized,dim))
        self.Cpn = CondPositional(height_patch=height//patch_size,
                                  width_patch=width//patch_size,
                                  dim=dim,
                                  color=None)
        self.color_cpn = CondPositional(height_patch=23,
                                  width_patch=23,
                                  dim=dim,
                                  color=q_ab)
        
        self.rebalance_loss = RebalanceLoss.apply
        self.get_class_weights = GetClassWeights(CIELAB(),
                                            lambda_=0.5)
        self.decode_q = AnnealedMeanDecodeQ(CIELAB(),T=0.38,device=device)


    def forward(self,img_L,img_ab,mask,training=False):
        #img_L B,1,H,W
        if img_L.size(1)==1:
            img = img_L.repeat(1,3,1,1)
        else:
            img = img_L.clone()
        encoder_out = self.encoder((img/100.))[:,1:,:] #B*N*dim exclude cls
        encoder_out = self.Cpn(encoder_out)
        colors_tokens = self.colors_tokens.expand(encoder_out.size(0),-1,-1)
        colors_tokens = self.color_cpn(colors_tokens,23,23)
        pred_q  = self.decoder(encoder_out,colors_tokens,mask)
        if img_ab is not None:
            soft_labels = encode(img_ab,self.sigma,self.neighbours,self.q_ab,self.bins)
        else:
            soft_labels = None
        if training:
            color_weights = self.get_class_weights(soft_labels)
            pred_q = self.rebalance_loss(pred_q, color_weights)
        pred_ab = self.decode_q(pred_q)
        final_img = torch.cat((img_L,pred_ab.detach()),dim=1)
        return pred_q,pred_ab,soft_labels,final_img
    
    


