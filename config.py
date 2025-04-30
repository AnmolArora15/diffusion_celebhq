import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256
    train_batch_size = 16
    eval_batch_size = 30
    num_epochs = 2
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_step = 250
    lr_scheduler_type:str = "linear"
    inference_steps = 50
    use_ema = False
    save_image_epochs = 100
    save_model_epochs = 500
    mixed_precision = "fp16"
    output_dir: str = "/scratch/aml_coursework/my_diffusion_project/samples/LDM"
    push_to_hub: bool = False
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed = 15
    train_dataset_path = '/scratch/aml_coursework/my_diffusion_project/data/celeba_hq_split/train'
    test_dataset_path = '/scratch/aml_coursework/my_diffusion_project/data/celeba_hq_split/test'
