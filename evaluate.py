import os
import time
import wandb
import torch
import math
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.utils.fid import compute_fid
from src.utils.make_grid import make_grid

def evaluate(config, epoch, pipeline, test_loader, device):
    pipeline.to(device)

    ####### Generating Fake Images ########
    fake_images = []
    total_images = 300
    num_batches = math.ceil(total_images / config.eval_batch_size)
    images_needed = total_images

    for i in range(num_batches):
        batch_size = min(config.eval_batch_size, images_needed)
        generator = torch.manual_seed(config.seed + i)  # Vary seed for each batch
        images = pipeline(batch_size=batch_size, generator=generator).images
        fake_images.extend([transforms.ToTensor()(img) for img in images])
        images_needed -= batch_size

    print(f'No. of fake images generated: {len(fake_images)}')
    fake_images = torch.stack(fake_images[:total_images]).to(device)  # Ensure exactly 300

    ####### Get Real Images ########
    real_images = []
    for batch in test_loader:
        images = batch[:config.eval_batch_size]  # Take only the image tensor if (images, labels)
        real_images.extend(images)
        if len(real_images) >= total_images:
            break
    real_images = torch.stack(real_images[:total_images]).to(device)

    ####### Compute FID ########
    fid_score = compute_fid(real_images, fake_images, device)
    print(f"FID Score: {fid_score}")

    ####### Save Generated Images ########
    fake_images_pil = [transforms.ToPILImage()(img.cpu().detach().clamp(-1, 1)) for img in fake_images[:16]]
    image_grid = make_grid(fake_images_pil, rows=4, cols=4)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    save_path = os.path.join(test_dir, f"{epoch:04d}_timestamp_{timestamp}.png")
    image_grid.save(save_path)
    print(f"Saved image grid to {save_path}")

    ####### Log to W&B ########
    wandb.log({
        f"Generated Images/Epoch {epoch}": wandb.Image(image_grid),
        # "FID Score": fid_score
    }, step=epoch)

    return fid_score
