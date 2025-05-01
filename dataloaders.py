from torch.utils.data import DataLoader
from dataset import CelebDataset
from torchvision import transforms
from config import TrainingConfig

def get_dataloaders(config):
    config = TrainingConfig()
    train_transform = transforms.Compose([
        transforms.Resize((config.image_size,config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.1,hue=0.05),
        transforms.RandomResizedCrop(128, scale=(0.9, 1.0),    # Minor zoom-ins / re-centering
                                  ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.image_size,config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = CelebDataset(config.train_dataset_path,transform=train_transform)
    test_dataset = CelebDataset(config.test_dataset_path, transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    return train_loader,test_loader