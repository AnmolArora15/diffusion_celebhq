import os
from torch.utils.data import Dataset
from PIL import Image

class CelebDataset(Dataset):
	def __init__(self,root_dir,transform=None):
		self.root_dir=root_dir
		self.transform= transform
		self.image_paths = sorted(os.listdir(root_dir))
		
	def __len__(self):
		return len(self.image_paths)
		
	def __getitem__(self,idx):
		img_path=os.path.join(self.root_dir,self.image_paths[idx])
		image = Image.open(img_path).convert('RGB')
		
		if self.transform:
			image = self.transform(image)
		
		return image
		
