
import random
import shutil
import numpy as np
import os
import tqdm
import torch
import scipy.io
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from compressai.entropy_models import GaussianConditional
import torch.nn.functional as F

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def dct_matrix(N):
    matrix = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                matrix[k, n] = np.sqrt(1/N)
            else:
                matrix[k, n] = np.sqrt(2/N) * np.cos(np.pi * (2*n + 1) * k / (2*N))
    return matrix

def eb_forward(x, eb, noise, means, scale, eval=False):  
    if eval:
        outputs = torch.round(x/noise.view(-1,1))
    else:
        outputs = (x)/noise.view(-1,1) + torch.empty_like(x).uniform_(-1/2, 1/2)


    means = means.view(1,-1).repeat(x.shape[0],1) / noise.view(-1,1)
    scale = scale.view(1,-1).repeat(x.shape[0],1) / (noise.view(-1,1)**2)

    likelihood = eb._likelihood(outputs, scale, means)
    if eb.use_likelihood_bound:
        likelihood = eb.likelihood_lower_bound(likelihood)
    return outputs, likelihood

class Model(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        dct = np.kron(dct_matrix(block_size), dct_matrix(block_size)).T
        eig = torch.from_numpy(dct).float()
        self.eig = nn.Parameter(eig)
        self.eb = GaussianConditional(None)
        self.means = nn.Parameter(torch.empty((block_size*block_size)).uniform_(-0.5,0.5))
        self.scale = nn.Parameter(torch.empty((block_size*block_size)).uniform_(0,1))
        self.quan =nn.Sequential(nn.Linear(1,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid())
        self.block_size=block_size
        
    def forward(self, x, eval=False, noise_idx=10, variable=True):
        coeff = torch.matmul(x,self.eig)
        
        if variable:
            noise = self.quan(noise_idx.view(-1,1)).squeeze()*50+10
        else:
            noise = noise_idx.view(-1)
        coeff_hat, likelihood = eb_forward(coeff.view(-1,self.block_size**2), self.eb, noise, self.means, self.scale, eval)
        coeff_hat = coeff_hat*noise.unsqueeze(-1)
        rec = torch.matmul(coeff_hat, self.eig.T)
        return coeff, rec, coeff_hat, likelihood
      
class MyDataset_residual(Dataset):
    def __init__(self, test=False, PM=0, block_size=8):
        super().__init__()
        import mat73
        if block_size==8:
            data_dict = mat73.loadmat('VideoDatasetBS8.mat')
            if test:
                self.img_patch = np.concatenate([data_dict['I_cols'][n][:,-data_dict['I_cols'][0].shape[-1]//6:].transpose(1,0).reshape(-1,64) for n in range(35)], axis=0)        # for k in np.arange(0,len(tiles)):
            else:
                self.img_patch = np.concatenate([data_dict['I_cols'][n][:,:data_dict['I_cols'][n].shape[-1]//6*5].transpose(1,0).reshape(-1,64) for n in range(35)], axis=0)        # for k in np.arange(0,len(tiles)):
        elif block_size==16:
            data_dict = scipy.io.loadmat('X_large_N=16.mat')
            if test:
                self.img_patch = data_dict['X'][:,:,data_dict['X'].shape[-1]//5*4:].transpose(2,0,1).reshape(-1,block_size**2)
            else:
                self.img_patch = data_dict['X'][:,:,:data_dict['X'].shape[-1]//5*4].transpose(2,0,1).reshape(-1,block_size**2)
        elif block_size==32:
            data_dict = scipy.io.loadmat('X_32x32_imagenet.mat')
            if test:
                self.img_patch = data_dict['X'][:,:,data_dict['X'].shape[-1]//5*4:].transpose(2,0,1).reshape(-1,block_size**2)
            else:
                self.img_patch = data_dict['X'][:,:,:data_dict['X'].shape[-1]//5*4].transpose(2,0,1).reshape(-1,block_size**2)
    
    def __len__(self):
        return len(self.img_patch)
    
    def __getitem__(self,idx):
        return torch.from_numpy(self.img_patch[idx]).float()

def save_checkpoint(state, base_dir, filename="checkpoint.pth.tar"):
    print(f"Saving checkpoint: {base_dir+filename}")
    torch.save(state, base_dir+filename)    

def train(dataloader, model, optimizer, epoch, lambda_max=0.05, lambda_min=0.001):
    model.train()
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader.dataset)//dataloader.batch_size)
    recon_loss_avg = AverageMeter()
    bpp_loss_avg = AverageMeter()
    rate_model_loss_avg = AverageMeter()
    rate_model_loss_l1_avg = AverageMeter()
    total_loss_avg = AverageMeter()
    aux_loss_avg = AverageMeter()
    psnr_avg = AverageMeter()
    for batch, (x) in progress_bar:
        noise_norm = []
        lambda_norm = []
        if epoch<200:
            lambda_norm = torch.tensor([0.1]*x.shape[0]).float()
            noise_norm = torch.tensor([1]*x.shape[0]).int()
            pred, rec, pred_noise,likelihoods = model(x,noise_idx=noise_norm, variable=False)
        else:
            for idx in range(x.shape[0]):
                lmbda_norm = random.random()
                lmbda_norm = (idx%x.shape[0])/x.shape[0]+lmbda_norm*(1/x.shape[0])
                lmbda = np.exp(lmbda_norm*(np.log(lambda_max)-np.log(lambda_min))+np.log(lambda_min))
                lambda_norm.append(lmbda)
                noise_norm.append(lmbda_norm)
            lambda_norm = torch.tensor(lambda_norm).float()
            noise_norm = torch.tensor(noise_norm).int()
            pred, rec, pred_noise,likelihoods = model(x,noise_idx=lambda_norm)
        recon_loss = nn.functional.mse_loss(x.view(x.shape[0],-1),rec.view(x.shape[0],-1),reduction='none').mean(-1)
        bpp_loss = torch.log(likelihoods).sum(-1) / (-math.log(2) * x.shape[-1])
        loss = lambda_norm*recon_loss+ bpp_loss
        recon_loss = recon_loss.mean()
        bpp_loss = bpp_loss.mean()
        loss = loss.mean()
        psnr_loss = 10 * torch.log10(255*255 / recon_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        recon_loss_avg.update(recon_loss.item(),n=len(x))
        bpp_loss_avg.update(bpp_loss.item(),n=len(x))
        psnr_avg.update(psnr_loss.item(),n=len(x))
        total_loss_avg.update(loss.item(),n=len(x))

        if batch%5==0:
            progress_bar.set_postfix_str(f"psnr: {psnr_avg.avg:.3f} recon_loss: {recon_loss_avg.avg:>4f} bpp_loss: {bpp_loss_avg.avg:>4f} total: {total_loss_avg.avg:>4f}")
            progress_bar.set_description_str(f"Epoch:{epoch}")


import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=8)
parser.add_argument('-lambda_max', type=float, default=0.5)
parser.add_argument('-lambda_min', type=float, default=0.01)
parser.add_argument('-exp_name', type=str, default='')
parser.add_argument('-lr', type=float, default=1e-2)
args = parser.parse_args(sys.argv[1:])

trained_eig = []
eng = None
EXP_NAME = f'{args.exp_name}'
base_dir = f'work/chiahaok/univ_transform/{EXP_NAME}/'
os.makedirs(base_dir, exist_ok=True)
model= Model(args.b)
model.train()
train_dataset = MyDataset_residual(block_size=args.b)
test_dataset = MyDataset_residual(block_size=args.b, test=True)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
optim = torch.optim.Adam(model.parameters(),lr=args.lr)
sceduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=list(range(50,200,50))+list(range(200,400,20)), gamma=0.5)
for epoch in tqdm.trange(400):
    train(train_dataloader, model, optim, epoch, args.lambda_max, args.lambda_min)
    sceduler.step()
    if epoch%50==49:
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
                "lr_scheduler": sceduler.state_dict(),
            },
            base_dir,
            filename=f'checkpoint.pth.tar'
        )
    if epoch%50==49:
        shutil.copyfile(base_dir+f'checkpoint.pth.tar', base_dir+ f"checkpoint_{epoch}.pth.tar")
trained_eig.append(model.eig.detach().numpy())
print('====================')
scipy.io.savemat(f'{args.exp_name}.mat',{'high':trained_eig[0]})