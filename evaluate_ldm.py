import os
import time
import wandb
import torch
import math
from torchvision import transforms
from src.utils.fid import compute_fid
from src.utils.make_grid import make_grid
from PIL import Image
from config import TrainingConfig

config = TrainingConfig()

@torch.no_grad()
def evaluate_ldm(config, epoch, unet, vae, noise_scheduler, test_loader, device):
    unet.eval()
    vae.eval()

    total_images = 300
    batch_size = config.eval_batch_size
    num_batches = math.ceil(total_images / batch_size)
    latents_list = []

    # === Generate 300 images ===
    for i in range(num_batches):
        bs = min(batch_size, total_images - len(latents_list))
        latents = torch.randn(bs, 4, 32, 32).to(device)
        noise_scheduler.set_timesteps(config.inference_steps)

        for t in noise_scheduler.timesteps:
            noise_pred = unet(latents, t).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents_list.append(latents)

    #  Decoding Generated latents
    all_latents = torch.cat(latents_list, dim=0)[:total_images]
    decoded_images = []
    for i in range(0, total_images, batch_size):
        latent_batch = all_latents[i:i + batch_size].to(device)
        with torch.amp.autocast('cuda'):  # optional fp16 decoding
            decoded_batch = vae.decode(latent_batch / 0.18215).sample
        decoded_batch = ((decoded_batch.clamp(-1, 1) + 1) / 2).cpu()
        decoded_images.append(decoded_batch)
        torch.cuda.empty_cache()
    decoded = torch.cat(decoded_images, dim=0).to(device)

    # === Get 300 real images ===
    real_images = []
    for batch in test_loader:
        real_images.extend(batch)
        if len(real_images) >= total_images:
            break
    real_images = torch.stack(real_images[:total_images])
    real_images = real_images.to(device)
    

    # === Compute FID ===
    fid_score = compute_fid(real_images, decoded, device)
    print(f"✅ FID Score (epoch {epoch}): {fid_score:.4f}")

    # === Save Grid ===
    grid_images = [transforms.ToPILImage()(img) for img in decoded[:16]]
    grid = make_grid(grid_images, rows=4, cols=4)

    save_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_dir, f"epoch_{epoch}_samples_{timestamp}.png")
    grid.save(save_path)

    # === Log to W&B ===
    wandb.log({
        f"Generated Images/Epoch {epoch}": wandb.Image(grid),
        "FID Score": fid_score
    }, step=epoch)

    return fid_score