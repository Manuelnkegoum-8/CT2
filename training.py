import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim
from transformer.model import CT2
from transformer.encoder import vit
from transformer.decoder import Decoder

from utils.datasets import *
from utils.trainer import *
from utils.quantization import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import sys
#import warmup_scheduler
import torch

from colorama import Fore, Style
import os,time
import torch.optim.lr_scheduler as lr_schedule
import argparse

rainbow_colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]

def printf(text='='*80,color=Fore.GREEN):
    for char in text:
        print(color + char, end="", flush=True)
        time.sleep(0.001)
    print('\n'+Style.RESET_ALL)
parser = argparse.ArgumentParser(description='Image Captioning on Flickr8k quick training script')

# Data args
parser.add_argument('--data_path', default='./data', type=str, help='dataset path')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
parser.add_argument('--freq', default=5, type=int, metavar='N', help='log frequency (by iteration)')

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
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup', default=5, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--beta', default=1., type=float, help='beta value for l1 smooth loss')
parser.add_argument('--weight_l1', default=10., type=float, help='weight for smooth l1 loss')
parser.add_argument('--resume', default=False, help='Version')
parser.add_argument('--seed', default=42, help='seed')
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
    device = torch.device("cuda")
    torch.cuda.manual_seed(44)
else:
    device = torch.device("cpu")

args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(44)
if __name__ == '__main__':
    init_process_group(backend="nccl") 
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    train_data, val_data = COCODataset(image_size=height,dataset_dir=data_location,split='train'), COCODataset(image_size=height,dataset_dir=data_location,split='val')
    train_sampler = DistributedSampler(train_data)
    val_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(
                    dataset=train_data,
                    batch_size=batch_size,
                    num_workers=args.workers,
                    shuffle=False,
                    sampler=train_sampler
                    )
    val_loader = DataLoader(
                    dataset=val_data,
                    batch_size=batch_size,
                    num_workers=args.workers,
                    shuffle=False,
                    sampler=train_sampler
                    )

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
    decoder = Decoder(encoder_dim=dim,
                      embed_dim=dim,
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
            device=device

    )
    
    model = model.to(device)
    model = model.to(local_rank)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9,nesterov=True)
    num_epochs = 0 
    final_epoch = args.epochs
    scheduler = lr_schedule.PolynomialLR(optimizer)
    criterion = nn.SmoothL1Loss(beta=beta_l1)
    weight_l1 = weight_l1
    sigma = sigma
    neighbours = neighbours
    bins = n_col_tokens 
    # Train the model
    best_loss = float('inf')
    torch.autograd.set_detect_anomaly(True)
    checkpoint_path = 'ct2.pt'
    if  os.path.exists(checkpoint_path):
        loc = f"cuda:{local_rank}"
        checkpoint = torch.load('ct2.pt',map_location=loc)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        num_epochs = checkpoint['epoch']+1

    
    model = DDP(model, device_ids=[local_rank])
    if global_rank == 0 :
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTrainable parameters: {trainable_params}")
        print(Fore.LIGHTGREEN_EX+'='*80)
        print("[INFO] Training for {0} epochs".format(final_epoch-num_epochs))
        print('='*80+Style.RESET_ALL)

    torch.cuda.synchronize()
    for epoch in range(num_epochs,final_epoch):
        train_loss = train_epoch(model,train_loader,optimizer,weight_l1,criterion,device)
        if (epoch+1)%args.freq==0:
            valid_loss,val_psnr = validate(model,val_loader,weight_l1,criterion,device)
            torch.cuda.synchronize()
            print(f"\n[GPU{global_rank}] | Epoch: {epoch+1} | Train Loss: {train_loss} | Val Loss: {valid_loss} | PSNR: {val_psnr}")
            if local_rank == 0:
                torch.save({
                            'model_state_dict': model.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                        }, f"ct2.pt")

    destroy_process_group()
    if global_rank == 0 :
        print(Fore.GREEN+'='*100)
        print("[INFO] End training")
        print('='*100+Style.RESET_ALL)