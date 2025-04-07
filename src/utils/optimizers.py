import torch
from torch.optim import AdamW
from config import TrainingConfig

def get_optimizer(model):
    config = TrainingConfig()

    optimizer = AdamW(model.parameters(),lr=config.learning_rate)

    return optimizer