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
from src.utils.ema import EMA

config = TrainingConfig()
os.makedirs(config.output_dir,exist_ok=True)

train_loader,test_loader = get_dataloaders(config=config)
unet = get_ldm_unet()
# checkpoint_path = os.path.join(config.output_dir,"ldm_unet_epoch_200.pt")

# if os.path.exists(checkpoint_path):
    # print(f"Loading checkpoint from {checkpoint_path}")
    # unet.load_state_dict(torch.load(checkpoint_path))

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

# EMA
if config.use_ema:
    ema = EMA(unet,decay=0.9999)
    ema.ema_model.to(unet.device)



accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="wandb",
    project_dir=config.output_dir
)

wandb.init(project='LDM Training', config = asdict(config))
# wandb.init(project='LDM Training',id="uoef9jcc",resume="must",config=asdict(config))
wandb.define_metric("epoch")
wandb.define_metric("FID (LDM)", step_metric="step")

vae.to(accelerator.device)

# Model Prep
unet,optimizer,train_loader,lr_scheduler=accelerator.prepare(unet,optimizer,train_loader,lr_scheduler)

#EMA
if config.use_ema:
    ema.ema_model.to(accelerator.device)

# Training Loop
global_step=0
vae.eval()

# start_epoch = 200
for epoch in range(config.num_epochs):
# for epoch in range(start_epoch,501):
    progress_bar = tqdm(total=len(train_loader),disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f'Epoch {epoch+1}')

    if config.use_ema:
        eval_unet=accelerator.unwrap_model(ema.get_model())
    else:
        eval_unet=accelerator.unwrap_model(unet)

    for step, batch in enumerate(train_loader):
        images = batch.to(accelerator.device)

        with torch.no_grad():
           latents = vae.encode(images).latent_dist.sample() * 0.18215
        #    print(latents.shape)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        with accelerator.accumulate(unet):
            noise_pred = unet(noisy_latents, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if config.use_ema:
                ema.update(unet)

        progress_bar.update(1)
        torch.cuda.empty_cache()

        wandb.log({
            "loss": loss.item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch+1,
            })
        
        
        global_step += 1
# FID Evaluation
    if accelerator.is_main_process:

        if (epoch + 1) % config.save_image_epochs == 0:
            print(f'Evaluating at epoch {epoch+1}')
            eval_unet = accelerator.unwrap_model(ema.get_model() if config.use_ema else unet)
            fid = evaluate_ldm(config, epoch+1,eval_unet, vae, noise_scheduler, test_loader, accelerator.device)
            wandb.log({"FID (LDM)": fid,
                     "epoch": epoch + 1,
                    "step": global_step, 
                    }, step=global_step)
            
# Saving model and genertaing images
    if accelerator.is_main_process:

        if (epoch + 1) % config.save_model_epochs == 0:
            model_path = os.path.join(config.output_dir, f"ldm_unet_epoch_{epoch+1}.pt")
            checkpoint = {
                "unet" : accelerator.unwrap_model(unet).state_dict(),
                "epoch": epoch,
                "global_step": global_step
            }
            if config.use_ema:
                checkpoint['ema_unet'] = ema.get_model().state_dict()

            torch.save(checkpoint,model_path)
            #wandb.save(model_path)

            with torch.no_grad():
                eval_unet.eval()
                latents = torch.randn(16, 4, 16, 16).to(accelerator.device)
                noise_scheduler.set_timesteps(config.inference_steps)

                for t in noise_scheduler.timesteps:
                    with torch.autocast(device_type=accelerator.device.type):
                        noise_pred = eval_unet(latents, t).sample
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                decoded = vae.decode(latents / 0.18215).sample
                decoded = (decoded.clamp(-1, 1) + 1) / 2
                
            