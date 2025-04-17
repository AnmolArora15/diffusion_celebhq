from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
from config import TrainingConfig

### LR Scheduler
def get_lr_scheduler(optimizer, train_loader):
    config = TrainingConfig()

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_step,
        num_training_steps=(len(train_loader) * config.num_epochs),
    )
    
    return lr_scheduler

#Linear Scheduler

from transformers import get_linear_schedule_with_warmup

def get_linear_scheduler(optimizer,train_loader):
    config = TrainingConfig()

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1*len(train_loader) * config.num_epochs),
        num_training_steps=(len(train_loader) * config.num_epochs)
    )
    return scheduler

# Noise Scheduler

def get_ddpm_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)

# Foward Diffusion (Adds Noise to images)
    noise = torch.randn(clean_images.shape).to(device)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device).long()
    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
    
    return noisy_images, noise, timesteps

     