from torch.utils.data import DataLoader
from dataset import CelebDataset
from torchvision import transforms
from config import TrainingConfig

def get_dataloaders(config):
    # config = TrainingConfig()

    transform = transforms.Compose([
        transforms.Resize((config.image_size,config.image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    train_dataset = CelebDataset(config.train_dataset_path,transform=transform)
    test_dataset = CelebDataset(config.test_dataset_path, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=True)

    return train_loader,test_loader