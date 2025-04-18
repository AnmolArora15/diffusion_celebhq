import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    eval_batch_size = 30
    num_epochs = 200
    gradient_accumulation_steps = 1
    learning_rate = 9e-5
    lr_warmup_step = 250
    lr_scheduler_type:str = "linear"
    use_ema = True
    save_image_epochs = 20
    save_model_epochs = 40
    mixed_precision = "fp16"
    output_dir: str = "/scratch/aml_coursework/my_diffusion_project/samples"
    push_to_hub: bool = False
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed = 15
    train_dataset_path = '/scratch/aml_coursework/my_diffusion_project/data/celeba_hq_split/train'
    test_dataset_path = '/scratch/aml_coursework/my_diffusion_project/data/celeba_hq_split/test'
