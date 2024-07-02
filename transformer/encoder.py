import torch
import  torch.nn as  nn
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
import numpy as np
import random,math
import torch.nn.functional as F

   

class patch_embedding(nn.Module):
    def __init__(self,height=224,width=224,n_channels=1,patch_size=16,dim=768):
        super().__init__()
        
        assert height%patch_size==0 and width%patch_size==0 ,"Height and Width should be multiples of patch size wich is {0}".format(patch_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_size = patch_size
        self.n_patchs = height*width//(patch_size**2)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patchs+1, dim))
        self.projection = nn.Conv2d(n_channels, dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        #x bs,h,w,c
        embedding = self.pos_embedding.to(x.device)
        #projection on the dim of model
        x = self.projection(x)
        bs,dim,h,w = x.size()
        x = x.permute(0,2,3,1).contiguous().view(bs,h*w,dim)
        cls_tokens = repeat(self.class_token, '() n d -> b n d', b = x.size(0))
        x = torch.cat((cls_tokens, x), dim=1)
        outputs = x + embedding[:,:x.size(1),:]
        return outputs
        
class LocallySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(LocallySelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tau = torch.sqrt(torch.tensor(self.head_dim))
        self.query  = nn.Linear(embed_dim, embed_dim,bias = True)
        self.key  = nn.Linear(embed_dim, embed_dim,bias = True)
        self.value  = nn.Linear(embed_dim, embed_dim,bias = True)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        #self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        self.query.bias.data.fill_(0)
        self.key.bias.data.fill_(0)
        self.value.bias.data.fill_(0)
        
    def _masked_softmax(self, attention_scores, attention_mask):
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float('-inf'))
        return F.softmax(attention_scores, dim=-1)

    def forward(self,x, need_weights=False, attn_mask=None):
        batch_size, seq_length, _ = x.size()
        f = lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads)
        q = f(self.query(x))
        k = f(self.key(x))
        v = f(self.value(x))
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.tau
        attention_scores = self._masked_softmax(attention_scores, attn_mask)
        attention_output = torch.matmul(attention_scores, v)
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        if need_weights:
            return self.o_proj(attention_output),attention_scores
        return self.o_proj(attention_output),None

class feedforward(nn.Module):
    def __init__(self,embed_dim=768,ff_hidden_dim=1024,dropout_rate = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )
    def forward(self, x):
        outputs = self.mlp(x)
        return outputs
        
class Transformer(nn.Module):
    def __init__(self, embed_dim=3, depth=4, heads=2, ff_hidden_dim=1024, dropout_rate = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(embed_dim),
                LocallySelfAttention(embed_dim,heads),
                nn.LayerNorm(embed_dim),
                feedforward(embed_dim,ff_hidden_dim, dropout_rate),
                
            ]))
    def forward(self, x,mask=None):
        for norm1,attn,norm2, ff in self.layers:
            x = attn(norm1(x),attn_mask = mask)[0] + x
            x = ff(norm2(x)) + x
        return x
        
class vit(nn.Module):
    def __init__(self,height=224,width=224,n_channels=3,patch_size=16,dim=512,n_head=2,feed_forward=1024,num_blocks=4):
        super().__init__()
        self.embedding = patch_embedding(height,width,n_channels,patch_size,dim)
        self.n_patchs = height*width//(patch_size**2)
        # Create a diagonal attention mask
        self.diag_attn_mask = torch.eye(self.n_patchs, dtype=torch.bool)
        self.transformer_encoder = Transformer(dim,num_blocks,n_head,feed_forward)
        self.num_blocks = num_blocks
    def forward(self, inputs):
        #x bs,h,w,c
        device_ = inputs.device
        x = self.embedding(inputs)
        outputs = self.transformer_encoder(x)
        return outputs

    def from_pretrained(self,path):
        new_state_dict = self.transformer_encoder.state_dict()
        key_mapping1 = {"class_token":"vit.embeddings.cls_token",
                        "pos_embedding":"vit.embeddings.position_embeddings",
                        "projection.weight":"vit.embeddings.patch_embeddings.projection.weight",
                        "projection.bias":"vit.embeddings.patch_embeddings.projection.bias",
                        }
        pretrained_state_dict = torch.load(path)
        embeddings_state_dict = self.embedding.state_dict()
        key_mapping2 = {}
        for i in range(self.num_blocks):
            key_mapping2.update({
                f"layers.{i}.1.query.weight": f"vit.encoder.layer.{i}.attention.attention.query.weight",
                f"layers.{i}.1.query.bias": f"vit.encoder.layer.{i}.attention.attention.query.bias",
                f"layers.{i}.1.key.weight": f"vit.encoder.layer.{i}.attention.attention.key.weight",
                f"layers.{i}.1.key.bias": f"vit.encoder.layer.{i}.attention.attention.key.bias",
                f"layers.{i}.1.value.weight": f"vit.encoder.layer.{i}.attention.attention.value.weight",
                f"layers.{i}.1.value.bias": f"vit.encoder.layer.{i}.attention.attention.value.bias",
                f"layers.{i}.1.o_proj.weight": f"vit.encoder.layer.{i}.attention.output.dense.weight",
                f"layers.{i}.1.o_proj.bias": f"vit.encoder.layer.{i}.attention.output.dense.bias",
                f"layers.{i}.3.mlp.0.weight": f"vit.encoder.layer.{i}.intermediate.dense.weight",
                f"layers.{i}.3.mlp.0.bias": f"vit.encoder.layer.{i}.intermediate.dense.bias",
                f"layers.{i}.3.mlp.3.weight": f"vit.encoder.layer.{i}.output.dense.weight",
                f"layers.{i}.3.mlp.3.bias": f"vit.encoder.layer.{i}.output.dense.bias",
                f"layers.{i}.2.weight": f"vit.encoder.layer.{i}.layernorm_after.weight",
                f"layers.{i}.2.bias": f"vit.encoder.layer.{i}.layernorm_after.bias",
                f"layers.{i}.0.weight": f"vit.encoder.layer.{i}.layernorm_before.weight",
                f"layers.{i}.0.bias": f"vit.encoder.layer.{i}.layernorm_before.bias"
            })
        print('loading pretrained weights for VIT')
        # Populate the new state dictionary with mapped weights
        for my_key,pretrained_key in key_mapping2.items():
            if pretrained_key in pretrained_state_dict:
                new_state_dict[my_key] = pretrained_state_dict[pretrained_key]
        for my_key,pretrained_key in key_mapping1.items():
            if pretrained_key in pretrained_state_dict:
                embeddings_state_dict[my_key] = pretrained_state_dict[pretrained_key]
        # Load the new state dictionary into your encoder
        self.transformer_encoder.load_state_dict(new_state_dict)
        self.embedding.load_state_dict(embeddings_state_dict)
