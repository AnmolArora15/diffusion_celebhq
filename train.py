import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from dataset import CelebDataset
from src.model import model
from config import TrainingConfig
from dataloaders import get_dataloaders
from src.utils.optimizers import get_optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from src.utils.scheduler import get_lr_scheduler, get_ddpm_scheduler
from evaluate import evaluate
from src.utils.fid import compute_fid
from src.utils.make_grid import make_grid

config = TrainingConfig()
optimizer = get_optimizer(model)
train_loader, test_loader = get_dataloaders(config=config)
lr_scheduler = get_lr_scheduler(optimizer,train_loader)
noise_scheduler = get_ddpm_scheduler()

# Accelerator
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="wandb",
    project_dir=config.output_dir
)

wandb.init(project="AML Project Diffusion", config=config.__dict__)

# Prepare for training
model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, lr_scheduler
)

global_step = 0

for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_loader):
        clean_images = batch
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        #################Forward Diffusion##############
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        wandb.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step})
        global_step += 1

    if accelerator.is_main_process:
        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        
        # Generate and Evaluate images 
        if (epoch + 1) % config.save_image_epochs == 0:
            print(f'Evaluating at epoch {epoch}')
            fid_score = evaluate(config, epoch, pipeline, test_loader, device=accelerator.device)
            wandb.log({"FID Score": fid_score, "Epoch": epoch}, step=epoch)

        if (epoch + 1) % config.save_model_epochs == 0:
            model_save_path = f"{config.output_dir}/model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path)
