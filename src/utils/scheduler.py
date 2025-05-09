from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from diffusers import DDPMScheduler,DDIMScheduler
from config import TrainingConfig

### LR Scheduler (cosine_schedule_with_warpup)
def get_lr_scheduler(optimizer, train_loader):
    config = TrainingConfig()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_step,
        num_training_steps=(len(train_loader) * config.num_epochs),
    )
    
    return lr_scheduler

#Linear Scheduler

def get_linear_scheduler(optimizer,train_loader):
    config = TrainingConfig()

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_step,
        num_training_steps=(len(train_loader) * config.num_epochs)
    )
    return scheduler

# Noise Scheduler

def get_ddpm_scheduler():
    return DDPMScheduler(num_train_timesteps=2000)



     