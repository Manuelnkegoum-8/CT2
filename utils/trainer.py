import torch,math
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
from skimage import color

def lab_to_rgb(img):
    return (255*np.clip(color.lab2rgb(img),0,1)).astype(np.uint8)

def PSNR(img1,img2):
    mse = (img1-img2)**2
    rmse = np.sqrt(np.mean(mse))
    psnr = 20*np.log10(255./rmse)
    return psnr


def eval_psnr(target_imgs,pred_imgs):
    psnr = 0.
    for target,pred in zip(target_imgs,pred_imgs):
        target = target.permute(1,2,0).detach().cpu().numpy()
        pred = pred.permute(1,2,0).detach().cpu().numpy()
        psnr += PSNR(target,pred)
    return psnr

def compute_loss(preds_q,pred_ab,soft_labels,img_ab,weight_l1=10.,criterion=None):
        
        B,C,h,w = preds_q.size()
        preds = preds_q.permute(0,2,3,1).contiguous().view(-1,C)
        probs = F.softmax(preds,dim=-1)
        soft_labels = soft_labels.permute(0,2,3,1).contiguous().view(-1,C)
        tmp = probs.clone()
        tmp2 = soft_labels.clone()
        
        probs = probs.masked_fill(tmp==0,1)
        log_probs = torch.log(probs)
        tmp2 = tmp2.masked_fill(soft_labels==0,1)
        labels = torch.log(tmp2)
        ll = (log_probs-labels)*soft_labels
        loss = -torch.sum(ll)/(B*h*w)
        l1_loss = criterion(pred_ab,img_ab/110.)
        return loss+weight_l1*l1_loss

def train_epoch(model,dataloader,optimizer,weight_l1,criterion,device):
    avg_loss = 0.0
    k = 0
    model.train()
    with tqdm(dataloader,unit='batch',ncols=80,colour='blue') as tepoch:
        for n,data in enumerate(tepoch):
            data = {key: value.to(device) for key, value in data.items()}
            img_L,img_ab,img_mask = data['L'], data['ab'],data['mask']
            bs = img_L.size(0)
            pred_q,soft_labels,out_ab,final_img = model(img_L,img_ab,img_mask,True)
            loss = compute_loss(pred_q,out_ab,soft_labels,img_ab,weight_l1,criterion)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
            avg_loss+=loss.item()*bs
            k+=bs
    return avg_loss/k

@torch.no_grad()
def validate(model,dataloader,weight_l1,criterion,device):
    avg_loss = 0.0
    avg_psnr = 0.0
    k = 0
    model.eval()
    with tqdm(dataloader,unit='batch',ncols=80,colour='green') as tepoch:
        for n,data in enumerate(tepoch):
            data = {key: value.to(device) for key, value in data.items()}
            img_L,img_ab,img_mask = data['L'], data['ab'],data['mask']
            bs = img_L.size(0)
            pred_q,soft_labels,out_ab,final_img = model(img_L,img_ab,img_mask)
            loss = compute_loss(pred_q,out_ab,soft_labels,img_ab,weight_l1,criterion)
            true_imgs = torch.cat((img_L,img_ab),dim=1)  
            psnr = eval_psnr(target_imgs=true_imgs,pred_imgs=final_img)
            avg_psnr += psnr
            tepoch.set_postfix(loss=loss.item())
            tepoch.set_postfix(psnr=psnr/bs)
            avg_loss+=loss.item()*bs
            k+=bs
    return avg_loss/k,avg_psnr/k