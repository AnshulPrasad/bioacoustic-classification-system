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
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.files = sorted(self.root_dir.rglob('*.png'))
        self. labels = self.labels_int()
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        logger.info("Total files(%s): %d", split, len(self.files))
        logger.info("Total labels(%s): %d", split, len(self.labels))

    def load_all_metadata(self):
        df = pd.concat(
            [pd.read_csv(f, usecols=['id', 'type']) for f in Path("../data/raw").glob("*.csv")],
            ignore_index=True
        )
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['type'])
        df['id'] = df['id'].astype(str)
        return df.set_index('id')['label']  # id → int label

    def labels_int(self):
        ids = [file.stem.split('_')[-3] for file in self.files]  # extract all IDs at once
        return self.load_all_metadata().reindex(ids, fill_value='None').tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.class_to_idx[path.name]
        image = Image.open(path).convert('RGB')
        return self.transform(image), label
