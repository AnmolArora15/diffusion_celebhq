import os
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb
from dataclasses import asdict
import torch.nn.functional as F
from diffusers import AutoencoderKL
from config import TrainingConfig
from dataloaders import get_dataloaders
from src.custom_vae import CustomVAE
from src.utils.optimizers import get_optimizer
from src.utils.scheduler import get_lr_scheduler, get_ddpm_scheduler, get_linear_scheduler
from src.custom_unet import get_ldm_unet
from evaluate_ldm import evaluate_ldm

config = TrainingConfig()
os.makedirs(config.output_dir,exist_ok=True)

train_loader,test_loader = get_dataloaders(config=config)
unet = get_ldm_unet()
optimizer = get_optimizer(model=unet)

if config.lr_scheduler_type == "linear":
    lr_scheduler = get_linear_scheduler(optimizer,train_loader)
elif config.lr_scheduler_type =="cosine":
    lr_scheduler = get_lr_scheduler(optimizer,train_loader)
else:
    raise ValueError(f'Unsupported LR Scheduler Type')

# vae = CustomVAE(latent_dim=4)
# vae.load_state_dict(torch.load('/scratch/aml_coursework/my_diffusion_project/samples/vae/vae_epoch_99.pt'))
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae.eval()

noise_scheduler = get_ddpm_scheduler()

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="wandb",
    project_dir=config.output_dir
)

wandb.init(project='LDM Training', config = asdict(config))

vae.to(accelerator.device)

# Model Prep
unet,optimizer,train_loader,lr_scheduler=accelerator.prepare(unet,optimizer,train_loader,lr_scheduler)

# Training Loop
global_step=0
vae.eval()

for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(train_loader),disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f'Epoch {epoch+1}')

    for step, batch in enumerate(train_loader):
        images = batch.to(accelerator.device)

        with torch.no_grad():
           latents = vae.encode(images).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0,noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        with accelerator.accumulate(unet):
            noise_pred = unet(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        torch.cuda.empty_cache()
        wandb.log({
            "loss": loss.item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch+1,
            })
        
        
        global_step += 1

    if accelerator.is_main_process:

        if (epoch + 1) % config.save_image_epochs == 0:
            print(f'Evaluating at epoch {epoch+1}')
            fid = evaluate_ldm(config, epoch+1, accelerator.unwrap_model(unet), vae, noise_scheduler, test_loader, accelerator.device)
            print(f'FID - {fid}')
            wandb.log({"FID (LDM)": fid}, step=epoch+1)
            
        
        # Generate and Evaluate images 
        if (epoch + 1) % config.save_model_epochs == 0:
            model_path = os.path.join(config.output_dir, f"ldm_unet_epoch_{epoch+1}.pt")
            torch.save(accelerator.unwrap_model(unet).state_dict(), model_path)
            wandb.save(model_path)

            with torch.no_grad():
                latents = torch.randn(16, 4, 8, 8).to(accelerator.device)
                noise_scheduler.set_timesteps(config.inference_steps)

                for t in noise_scheduler.timesteps:
                    noise_pred = unet(latents, t).sample
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                decoded = vae.decode(latents / 0.18215).sample
                decoded = (decoded.clamp(-1, 1) + 1) / 2
                wandb.log({"Generated Samples": [wandb.Image(img) for img in decoded]}, step=epoch+1)

            