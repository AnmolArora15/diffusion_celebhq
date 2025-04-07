import os
import time
import wandb
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import transforms
from PIL import Image
from src.utils.fid import compute_fid
from src.utils.make_grid import make_grid

def evaluate(config,epoch,pipeline,test_loader,device):
    pipeline.to(device)

    #######Generating Fake Images########
    fake_images=[]
    num_batches = 300//config.eval_batch_size
    for _ in range(num_batches):
        images = pipeline(batch_size=config.eval_batch_size, generator=torch.manual_seed(config.seed)).images
        fake_images.extend([transforms.ToTensor()(img) for img in images])
    
    print(f'No. of fake images generated : {len(fake_images)}')
    fake_images = torch.stack(fake_images).to(device)

    ### Get Real Images ###
    real_images = []
    for batch in test_loader:
        real_images.extend(batch[:config.eval_batch_size])  # Taking images only, not labels
    real_images = torch.stack(real_images).to(device)

    ### Compute FID ###
    fid_score = compute_fid(real_images, fake_images, device)
    print(f"FID Score: {fid_score}")

    ### Save Generated Images ###
    fake_images_pil = [transforms.ToPILImage()(img.cpu().detach().clamp(-1, 1)) for img in fake_images[:16]]
    image_grid = make_grid(fake_images_pil, rows=4, cols=4)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    save_path = f"{test_dir}/{epoch:04d}_timestamp_{timestamp}.png"
    print(f"Saving images to {test_dir}")
    image_grid.save(save_path)

    ### Log to W&B ###
    wandb.log({
        f"Generated Images/Epoch {epoch}": wandb.Image(image_grid),
        "FID Score": fid_score
    }, step=epoch)

    return fid_score