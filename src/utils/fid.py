import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

def compute_fid(real_images, fake_images, device):

    real_images = F.interpolate(real_images, size=(299, 299), mode="bilinear", align_corners=False)
    fake_images = F.interpolate(fake_images, size=(299, 299), mode="bilinear", align_corners=False)

    real_images = ((real_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    fake_images = ((fake_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # Computing FID between Generates and real images
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    return fid.compute().item()