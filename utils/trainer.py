import torch,math
from tqdm import tqdm
import torch.nn.functional as F


def compute_loss(preds_q,pred_ab,soft_labels,img_ab,weight_l1=10.,criterion=None):
        B,C,h,w = preds_q.size()
        preds_q = F.softmax(preds_q.permute(0,2,3,1).view(-1,C),dim=-1)
        soft_labels = soft_labels.permute(0,2,3,1).view(-1,C)
        tmp = preds_q.clone()
        tmp2 = soft_labels.clone()
        loss = (torch.log(preds_q.masked_fill(tmp==0,1))-torch.log(soft_labels.masked_fill(tmp2==0,1)))*soft_labels
        loss = -loss.sum()/(B*h*w)
        l1_loss = criterion(pred_ab,img_ab)
        #print("\n",pred_ab)
        print("\n",loss)
        print("\n",l1_loss)
        return loss+weight_l1*l1_loss

def train_epoch(model,dataloader,optimizer,weight_l1,criterion,device):
    avg_loss = 0.0
    model.train()
    with tqdm(dataloader,unit='batch',ncols=80,colour='blue') as tepoch:
        for n,data in enumerate(tepoch):
            data = {key: value.to(device) for key, value in data.items()}
            img_L,img_ab,img_mask = data['L'], data['ab'],data['mask']
            bs = img_L.size(0)
            pred_q,pred_ab,soft_labels,final_img = model(img_L,img_ab,img_mask,True)
            loss = compute_loss(pred_q,pred_ab,soft_labels,img_ab,weight_l1,criterion)
            loss = loss/16
            
            loss.backward()
            if n%16==0:
                optimizer.step()
                optimizer.zero_grad()
                tepoch.set_postfix(loss=loss.item())
            avg_loss+=loss.item()*bs
    return avg_loss/(n+1)

@torch.no_grad()
def validate(model,dataloader,weight_l1,criterion,device):
    avg_loss = 0.0
    model.eval()
    with tqdm(dataloader,unit='batch',ncols=80,colour='green') as tepoch:
        for n,data in enumerate(tepoch):
            data = {key: value.to(device) for key, value in data.items()}
            img_L,img_ab,img_mask = data['L'], data['ab'],data['mask']
            bs = img_L.size(0)
            pred_q,pred_ab,soft_labels,final_img = model(img_L,img_ab,img_mask)
            loss = compute_loss(pred_q,pred_ab,soft_labels,img_ab,weight_l1,criterion)
            tepoch.set_postfix(loss=loss.item())
            avg_loss+=loss.item()*bs
    return avg_loss/(n+1)