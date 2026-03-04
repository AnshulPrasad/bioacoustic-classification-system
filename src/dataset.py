# Responsibilities:
# - PyTorch Dataset class for loading spectrograms
# - DataLoaders for train/val/test
# - Label encoding

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class BirdSoundDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.classes = sorted([d.name for d in self.root_dir.iterdir()])
        self.class_to_idx = {c:i for i, c in enumerate(self.classes)}
        self.files = list(self.root_dir.glob('*.png'))
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.class_to_idx[path.name]
        image = Image.open(path).convert('RGB')
        return self.transform(image), label
