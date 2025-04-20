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
from src.utils.scheduler import get_lr_scheduler, get_ddpm_scheduler, get_linear_scheduler
from evaluate import evaluate
from src.utils.fid import compute_fid
from src.utils.make_grid import make_grid
from src.utils.ema import EMA

config = TrainingConfig()
optimizer = get_optimizer(model)
train_loader, test_loader = get_dataloaders(config=config)
noise_scheduler = get_ddpm_scheduler()

if config.lr_scheduler_type == "linear":
    lr_scheduler = get_linear_scheduler(optimizer,train_loader)
elif config.lr_scheduler_type =="cosine":
    lr_scheduler = get_lr_scheduler(optimizer,train_loader)
else:
    raise ValueError(f'Unsupported LR Scheduler Type')

# Accelerator
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="wandb",
    project_dir=config.output_dir
)

wandb.init(project="AML Project Diffusion", config=config.__dict__)

if config.use_ema:
    ema = EMA(model,decay=0.9999)

# Prepare for training
model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, lr_scheduler
)

if config.use_ema:
    ema.ema_model.to(accelerator.device)
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
            if config.use_ema:
                ema.update(model)

        progress_bar.update(1)
        torch.cuda.empty_cache()
        wandb.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step})
        global_step += 1

    if accelerator.is_main_process:
        # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        ema_pipeline = DDPMPipeline(unet=accelerator.unwrap_model(ema.get_model()), scheduler = noise_scheduler)  #Using EMA model for evaluation
        
        # Generate and Evaluate images 
        if (epoch + 1) % config.save_image_epochs == 0:
            print(f' Evaluating at epoch {epoch} (EMA model)')
            fid_score = evaluate(config, epoch, ema_pipeline, test_loader, device=accelerator.device)
            wandb.log({"FID Score (EMA)": fid_score, "Epoch": epoch}, step=epoch)

        if (epoch + 1) % config.save_model_epochs == 0:
            model_save_path = f"{config.output_dir}/model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), model_save_path)
            ema_model_save_path = f"{config.output_dir}/ema_model_epoch_{epoch}.pt"
            torch.save(ema.get_model().state_dict(), ema_model_save_path)
            wandb.save(ema_model_save_path)
            wandb.save(model_save_path)
