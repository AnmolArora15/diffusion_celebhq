
import os
import torch

import torch.nn.functional as F
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb
from config import TrainingConfig
from dataset import CelebDataset
from dataloaders import get_dataloaders
from src.utils.optimizers import get_optimizer
from src.utils.scheduler import get_lr_scheduler, get_ddpm_scheduler, get_linear_scheduler
from evaluate import evaluate
from src.utils.ema import EMA
from dataclasses import asdict
from src.dit_xl import DiT_XL_2, DiT_S_4,DiT_L_4
from diffusers import DDPMPipeline 
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

config = TrainingConfig()
model = DiT_L_4(input_size=32, in_channels=4, num_classes=0,learn_sigma=False)
print(f'model config :{model.config}')
# model = DiT_XL_2(input_size=256, in_channels=3, num_classes=0)
ckpt = torch.load("checkpoints/DiT-XL-2-256x256.pt", map_location="cpu")
model_state = model.state_dict()
filtered_ckpt = {
    k: v for k, v in ckpt.items()
    if k in model_state and v.shape == model_state[k].shape
}
model.load_state_dict(filtered_ckpt, strict=False)

print("Successfully loaded compatible weights.")

if os.path.exists(config.resume_path):
    model.load_state_dict(torch.load(config.resume_path, map_location="cpu"))
    print(f"Resumed from {config.resume_path}")

vae = AutoencoderKL.from_pretrained("/scratch/aml_coursework/my_diffusion_project/vae_weights")

optimizer = get_optimizer(model)

train_loader, test_loader = get_dataloaders(config=config)

noise_scheduler = get_ddpm_scheduler()

lr_scheduler = (
    get_linear_scheduler(optimizer, train_loader)
    if config.lr_scheduler_type == "linear"
    else get_lr_scheduler(optimizer, train_loader)
)

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="wandb",
    project_dir=config.output_dir,
)

wandb.init(project="DiT", config=asdict(config))

vae.to(accelerator.device)
vae.eval()
# Disable gradients
for param in vae.parameters():
    param.requires_grad = False

model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, lr_scheduler
)

if config.use_ema:
    ema = EMA(model, decay=0.9999)
    ema.ema_model.to(accelerator.device)

# Training
global_step = 0
for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_loader):
        clean_images = batch
        with torch.no_grad():
            latents = vae.encode(clean_images.to(accelerator.device)).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        bs = latents.size(0)

        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device
        ).long()
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            noise_pred = model(noisy_latents, timesteps,y=None)
            # predicted_noise = noise_pred[:, :3, :, :]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if config.use_ema:
                ema.update(model)

        progress_bar.update(1)
        wandb.log({
            "loss": loss.item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step
        })
        global_step += 1

# Evaluation 
    if accelerator.is_main_process and (epoch + 1) % config.save_image_epochs == 0:
        print(f"Evaluating at Epoch {epoch} - EMA Model")
        ema_model = accelerator.unwrap_model(ema.get_model()).to(accelerator.device)
        ema_model.eval()

        num_samples = 16
        sample_dir = f"{config.output_dir}/samples_epoch_{epoch}"
        os.makedirs(sample_dir, exist_ok=True)

        x = torch.randn(num_samples, 4, 32,32, device=accelerator.device)
        for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
            t_tensor = torch.tensor(t, device=accelerator.device).long()
            with torch.no_grad():
                noise_pred = ema_model(x, t_tensor.expand(x.size(0)), y=None)
                # predicted_noise = noise_pred[:, :3, :, :] 
            x = noise_scheduler.step(noise_pred, t_tensor, x).prev_sample

        samples = vae.decode(x / 0.18215).sample
        samples = (samples.clamp(-1, 1) + 1) / 2
        save_image(samples, f"{sample_dir}/samples.png", nrow=4)

        wandb.log({
            f"Sample Grid Epoch {epoch}": wandb.Image(samples, caption=f"Epoch {epoch} Samples")
        }, step=epoch)
        

    if accelerator.is_main_process and (epoch + 1) % config.save_model_epochs == 0:
        torch.save(model.state_dict(), f"{config.output_dir}/model_epoch_{epoch}.pt")
        if config.use_ema:
            torch.save(ema.get_model().state_dict(), f"{config.output_dir}/ema_model_epoch_{epoch}.pt")

