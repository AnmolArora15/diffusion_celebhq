import torch
from diffusers import DDPMPipeline, DDIMScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from config import TrainingConfig
from src.model import model as unet_model
from src.utils.make_grid import make_grid
from src.utils.fid import compute_fid
from dataloaders import get_dataloaders
from torchvision import transforms
from PIL import Image
import os

# === Configs ===
config = TrainingConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ema_ckpt_path = f'{config.output_dir}/ema_model_epoch_399.pt'
output_dir = os.path.join(config.output_dir, "sampling_comparison")
os.makedirs(output_dir, exist_ok=True)

# === Load EMA Model ===
model = unet_model
model.load_state_dict(torch.load(ema_ckpt_path, map_location=device))
model.to(device).eval()

# PNDM
class SafePNDMScheduler(PNDMScheduler):
    def step(self, model_output, timestep, sample, **kwargs):
        kwargs.pop("generator", None)
        return super().step(model_output, timestep, sample, **kwargs)

# === Sampler Setup ===
schedulers = {
    "DDIM": DDIMScheduler(num_train_timesteps=1000),
    # "PNDM": SafePNDMScheduler(num_train_timesteps=1000),
    # "DPM-Solver": DPMSolverMultistepScheduler(num_train_timesteps=1000)
}

# === DataLoader for real images ===
_, test_loader = get_dataloaders(config)

# === Sampling Config ===
num_images = 300
num_steps = 250
rows, cols = 4, 4  # for visualization

# === Loop Through Samplers ===
for name, scheduler in schedulers.items():
    print(f'\n--- Sampling with {name} ---')

    # Prepare pipeline
    pipeline = DDPMPipeline(unet=model, scheduler=scheduler).to(device)
    scheduler.set_timesteps(num_steps)

    generator = torch.manual_seed(config.seed)
    fake_images = []

    # Generate 300 images in batches
    batch_size = config.eval_batch_size
    num_batches = num_images // batch_size
    remaining = num_images % batch_size

    for i in range(num_batches + (1 if remaining > 0 else 0)):
        current_bs = batch_size if i < num_batches else remaining
        if current_bs == 0:
            continue
        images = pipeline(batch_size=current_bs, generator=generator, num_inference_steps=num_steps).images
        fake_images.extend([transforms.ToTensor()(img) for img in images])

    fake_images = torch.stack(fake_images[:num_images]).to(device)
    print(f"Generated {len(fake_images)} images for {name}")

    # === Load 300 real images ===
    real_images = []
    for batch in test_loader:
        real_images.extend(batch)
        if len(real_images) >= num_images:
            break
    real_images = torch.stack(real_images[:num_images]).to(device)

    # === Compute FID ===
    fid_score = compute_fid(real_images, fake_images, device)
    print(f"✅ {name} FID: {fid_score:.4f}")

    # === Save grid preview ===
    preview = [transforms.ToPILImage()(img.clamp(-1, 1)) for img in fake_images[:rows * cols]]
    grid = make_grid(preview, rows=rows, cols=cols)
    grid.save(os.path.join(output_dir, f"samples_{name.lower().replace(' ', '_')}.png"))

    # === Save FID to text file ===
    with open(os.path.join(output_dir, f"fid_{name.lower()}.txt"), "w") as f:
        f.write(f"{name} FID: {fid_score:.4f}\n")

