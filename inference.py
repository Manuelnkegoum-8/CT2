import numpy as np
import torch
import torch.optim
from transformer.model import CT2
from transformer.encoder import vit
from transformer.decoder import Decoder
from utils.quantization import *
import gradio as gr
from skimage import color
import pickle
import torchvision.transforms as transforms
import torch
from PIL import Image
import os
import argparse
from transformer.model import encode

parser = argparse.ArgumentParser(description='Image Captioning on Flickr8k quick training script')

# Data args
parser.add_argument('--data_path', default='./data', type=str, help='dataset path')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--freq', default=10, type=int, metavar='N', help='log frequency (by iteration)')

# Model parameters
parser.add_argument('--height', default=224, type=int, metavar='N', help='image height')
parser.add_argument('--width', default=224, type=int, metavar='N', help='image width')
parser.add_argument('--channel', default=3, type=int, help='disable cuda')
parser.add_argument('--cls', default=313, type=int, metavar='N', help='quantized filtered colors tokens')
parser.add_argument('--enc_heads', default=12, type=int, help='number of encoder  heads')
parser.add_argument('--enc_depth', default=12, type=int, help='number of encoder blocks')
parser.add_argument('--dec_heads', default=12, type=int, help='number of decoder  heads')
parser.add_argument('--dec_depth', default=2, type=int, help='number of decoder blocks')
parser.add_argument('--patch_size', default=16, type=int, help='patch size')
parser.add_argument('--dim', default=768, type=int, help='embedding dim of patch')
parser.add_argument('--enc_mlp_dim', default=3072, type=int, help='feed forward hidden_dim for an encoder block')
parser.add_argument('--dec_mlp_dim', default=3072, type=int, help='feed forward hidden_dim for a decoder block')
parser.add_argument('--neighbours', default=5, type=int, help='neighbours qauntized ab')
parser.add_argument('--sigma', default=5., type=float, help='std in gaussian kernel weighting for soft encoded labels')
parser.add_argument('--upsample', default=4, type=int, help='num_upsampling blocks in decoder')


# Optimization hyperparams
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup', default=5, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--beta', default=1., type=float, help='beta value for l1 smooth loss')
parser.add_argument('--weight_l1', default=10., type=float, help='weight for smooth l1 loss')
parser.add_argument('--resume', default=False, help='Version')

args = parser.parse_args()
data_location = args.data_path
lr = args.lr
weight_decay = args.weight_decay
height, width, n_channels = args.height, args.width, args.channel
patch_size, dim, enc_head = args.patch_size, args.dim, args.enc_heads
enc_feed_forward, enc_depth = args.enc_mlp_dim, args.enc_depth
dec_feed_forward, dec_depth = args.dec_mlp_dim, args.dec_depth
dec_head = args.dec_heads
num_upsampling = args.upsample
beta_l1 = args.beta
weight_l1 = args.weight_l1
batch_size = args.batch_size
warmup = args.warmup
neighbours = args.neighbours
sigma = args.sigma
n_col_tokens = args.cls

device = ""
if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

encoder = vit(height=height,
                  width=width,
                  n_channels=n_channels,
                  patch_size=patch_size,
                  dim=dim,
                  n_head=enc_head,
                  feed_forward=enc_feed_forward,
                  num_blocks=enc_depth
                  )
encoder.from_pretrained('vit_pretrained.bin')
decoder = Decoder(embed_dim=dim,
                      ff_hidden_dim=dec_feed_forward,
                      height=height,
                      width=width,
                      patch_size=patch_size,
                      num_heads=dec_head,
                      num_upsampling=num_upsampling)

q_ab = CIELAB().q_to_ab
model = CT2(
            encoder=encoder,
            decoder=decoder,
            height=height,
            width=width,
            patch_size=patch_size,
            dim=dim,
            neighbours=neighbours,
            num_quantized=n_col_tokens,
            sigma=sigma,
            q_ab=q_ab,
            gamma_l1=beta_l1,
            weight_l1=weight_l1,

    )
state_dict = torch.load('ct2.pt',map_location='cpu')
model.load_state_dict(state_dict['model_state_dict'])
model = model.to(device)
model.eval()
img_transform = transforms.Compose([
                transforms.Resize((height,width)),
            ])
mask_num = 4
fp = open(os.path.join('./', 'mask_prior.pickle'), 'rb')
L_dict = pickle.load(fp)
mask_L = np.zeros((mask_num, 313)).astype(np.bool_)     # [4, 313]
for key in range(101):
    for ii in range(mask_num):
        start_key = ii * (100//mask_num)      # 0
        end_key = (ii+1)* (100//mask_num)     # 25
        if start_key <= key < end_key:
            mask_L[ii, :] += L_dict[key].astype(np.bool_)
            break
mask_L = mask_L.astype(np.float32)
def lab_to_rgb(img):
    return 255*np.clip(color.lab2rgb(img),0,1).astype(np.uint8)
@torch.no_grad()
def predict(img):
    img = Image.fromarray(img)
    img = img_transform(img)
    img = np.array(img)
    img = color.rgb2lab(img).astype(np.float32)
    mask_p_c = np.zeros((height*width, n_col_tokens), dtype=np.float32)
    l = img[:,:,0].reshape((height*width))
    for l_range in range(4):
        start_l1, end_l1 = l_range * (100//mask_num), (l_range + 1) * (100 // mask_num)
        if end_l1 == 100:
            index_l1 = np.where((l >= start_l1) & (l <= end_l1))[0]
        else:
            index_l1 = np.where((l >= start_l1) & (l < end_l1))[0]
        mask_p_c[index_l1, :] = mask_L[l_range, :]

    mask_p_c = mask_p_c.reshape((height,width,n_col_tokens))
    mask = torch.from_numpy(mask_p_c).to(device).unsqueeze(0)
    img_L = torch.from_numpy(img[:,:,0]).type(torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    pp,_,_,res = model(img_L,None,mask,False)
    out = res[0].permute(1,2,0).detach().cpu().numpy()
    return color.lab2rgb(out)
    
with gr.Blocks() as demo:
    gr.Markdown(f"Colorization")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(image_mode='RGB',)
            btn = gr.Button(value="Process")
        with gr.Column():    
            out = gr.Image()

    btn.click(fn=predict, inputs=inp, outputs=out)

if __name__ == '__main__':
    demo.launch()

