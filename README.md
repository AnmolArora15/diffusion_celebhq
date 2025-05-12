# Unconditional Diffusion Model – AML Coursework

This project trains an unconditional image generation model using a diffusion process. It uses a UNet-based architecture and follows the DDPM framework. The model is trained on the CelebA dataset using 2700 training and 300 test images. FID (Fréchet Inception Distance) is used to evaluate performance.

---

## Files and Their Purpose:
- `train.py`                : Main training loop.
- `evaluate.py`            : Generates 300 fake images and computes FID with 300 test images.
- `config.py`              : Contains all training hyperparameters like batch size, learning rate, etc.
- `dataloaders.py`         : Prepares training and test data.
- `src/model.py`           : Defines the UNet model architecture.
- `src/utils/`             : Includes helper code for EMA, FID computation, optimizers, learning rate schedulers, and image grid generation.

---

## Requirements:

This project uses the following Python libraries:
- `torch`
- `torchvision`
- `diffusers`
- `accelerate`
- `wandb`
- `tqdm`
- `Pillow`

It's recommended to use a virtual environment or Google Colab.

---

## How to Run:
1. Clone the Repository
   ```bash
   git clone https://github.com/AnmolArora15/diffusion_celebhq.git
   cd diffusion_celebhq

2. Make sure the CelebA dataset is downloaded and paths are correctly set in `config.py`.

3. Model Training:
   ```bash
	python train.py
- Starts training from scratch (Epoch 0) using settings from `config.py`.
- Saves model checkpoints and 4×4 image grids every few epochs.
- Logs training loss and FID to Weights & Biases (W&B).

4. Evaluation is automatically handled in `train.py` by calling `evaluate.py` every `save_image_epochs`:
   - Uses EMA model to generate 300 fake images.
   - Computes FID against 300 real test images.
   - Saves images and model checkpoints.

5. Check different Hyperparameter in `config.py` as change it accordingly.

6. Sampling with different Schedulers
   ```bash
   	sampling_with_Scheduler.py
 
- Set `ema_chpt_path` to your saved EMA model checkpoint.
- The script loads the model and uses schedulers like DDIM, PNDM, and DPM-Solver to generate 300 images.
- Images are saved and evaluated using FID.
- Helps visually and quantitatively compare different samplers.

---

## NOTE:
Always check training configuration in `config.py` before starting training to ensure the parameters are set as intended.
