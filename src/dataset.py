# Responsibilities:
# - PyTorch Dataset class for loading spectrograms
# - DataLoaders for train/val/test
# - Label encoding

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from logger import get_logger
logger = get_logger(__name__, 'dataset.log')

class BirdSoundDataset(Dataset):
    def __init__(self, SPLIT_DIR, RAW_DIR, split='train', transform=None):
        self.SPLIT_DIR = Path(SPLIT_DIR) / split
        self.RAW_DIR = RAW_DIR
        self.split = split
        self.files = sorted(self.SPLIT_DIR.rglob('*.png'))
        self. labels = self.labels_int()
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def load_all_metadata(self):
        logger.info(f'Loading all metadata({self.split})')
        df = pd.concat([pd.read_csv(f, usecols=['id', 'type']) for f in sorted(Path(self.RAW_DIR).glob("*.csv"))],ignore_index=True)
        df['id'] = df['id'].astype(str)
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['type'])
        self.num_classes = df['label'].max() + 1
        return df.set_index('id')['label']  # id → int label

    def labels_int(self):
        logger.info(f'Loading labels({self.split})')
        ids = [file.stem.split('_')[-3] for file in self.files]  # extract all IDs at once
        logger.info('Loaded labels')
        return self.load_all_metadata().reindex(ids, fill_value=-1).tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image), self.labels[idx]
