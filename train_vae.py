import os
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb
import torchvision
from dataclasses import asdict
from config import TrainingConfig
from dataloaders import get_dataloaders
from src.custom_vae import CustomVAE
from src.utils.vae_loss import vae_loss
from src.utils.optimizers import get_optimizer
from torchvision.utils import make_grid, save_image

config = TrainingConfig()
os.makedirs(config.output_dir,exist_ok=True)
vae = CustomVAE(latent_dim=4)
optimizer = get_optimizer(vae)
train_loader,test_loader = get_dataloaders(config=config)

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with='wandb',
    project_dir= config.output_dir
)

wandb.init(project="AML Project VAE", config=asdict(config))

code_artifact = wandb.Artifact("VAE-training-code", type="code")
code_artifact.add_file("config.py")
code_artifact.add_file("src/custom_vae.py")
wandb.log_artifact(code_artifact)

# Prepare for training
vae, optimizer, train_loader = accelerator.prepare(vae, optimizer, train_loader)

vae.train()
global_step = 0

for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_loader):
        images = batch.to(accelerator.device)

        with accelerator.accumulate(vae):
            recon, mu, logvar = vae(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        wandb.log({
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "step": global_step
        })
        global_step += 1

    if accelerator.is_main_process and (epoch + 1) % config.save_image_epochs == 0:
        vae.eval()
        with torch.no_grad():
            originals = images[:8] * 0.5 + 0.5
            recon, _, _ = vae(images[:8])
            recon = recon * 0.5 + 0.5  # Unnormalize
            recon = recon.clamp(0,1)
            orignals = originals.clamp(0,1)

            combined = torch.cat([originals, recon], dim=0)
            grid = make_grid(combined, nrow=8)

            save_path = os.path.join(config.output_dir, f"comparison_epoch_{epoch}.png")
            save_image(grid, save_path)

            wandb.log({
                "Comparison Grid": wandb.Image(grid)
        }, step=epoch)
    vae.train()
           

    if accelerator.is_main_process and (epoch + 1) % config.save_model_epochs == 0:
        model_save_path = os.path.join(config.output_dir, f"vae_epoch_{epoch}.pt")
        torch.save(accelerator.unwrap_model(vae).state_dict(), model_save_path)
        wandb.save(model_save_path)

