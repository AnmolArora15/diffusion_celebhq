import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256
    train_batch_size = 16
    eval_batch_size = 30
    num_epochs = 500
    gradient_accumulation_steps = 2
    learning_rate = 0.0001
    lr_warmup_step = 300
    lr_scheduler_type:str = "linear"
    inference_steps = 300
    use_ema = True
    save_image_epochs = 50
    save_model_epochs = 500
    mixed_precision = "fp16"
    output_dir: str = "/scratch/aml_coursework/my_diffusion_project/samples/LDM"
    push_to_hub: bool = False
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True
    seed = 42
    train_dataset_path = '/scratch/aml_coursework/my_diffusion_project/data/celeba_hq_split/train'
    # train_dataset_path = '/scratch/aml_coursework/my_diffusion_project/data/CelebDataset'
    test_dataset_path = '/scratch/aml_coursework/my_diffusion_project/data/celeba_hq_split/test'
