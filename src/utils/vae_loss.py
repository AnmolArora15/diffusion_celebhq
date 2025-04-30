import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(recon,x,mu,logvar):
    recon_loss = F.mse_loss(recon,x,reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    vae_loss = recon_loss + 1e-3 * kl_loss, recon_loss, kl_loss
    return vae_loss