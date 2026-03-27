# Responsibilities:
# - PyTorch Dataset class for loading spectrograms
# - DataLoaders for train/val/test
# - Label encoding

import json
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
        self.SPECTROGRAM_DIR = SPECTROGRAM_DIR
        self.files = list(Path(self.SPECTROGRAM_DIR).rglob("*.png"))
        self.train_paths, self.val_paths, self.test_paths = self.split_dataset()
        self.train_labels = self.encode(self.train_paths)
        self.val_labels = self.encode(self.val_paths)
        self.test_labels = self.encode(self.test_paths)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def build_and_save_index(self, index_path):
        data = {
            "train_paths": [str(p) for p in self.train_paths],
            "val_paths": [str(p) for p in self.val_paths],
            "test_paths": [str(p) for p in self.test_paths],
            "train_labels": self.train_labels,
            "val_labels": self.val_labels,
            "test_labels": self.test_labels,
            "num_classes": int(self.num_classes),
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def id_label(self):
        # build recording_id → label mapping from CSVs
        dfs = [pd.read_csv(f, usecols=['id', 'type']) for f in sorted(Path(self.RAW_DIR).glob("*.csv"))]
        df = pd.concat(dfs,ignore_index=True)
        df['id'] = df['id'].astype(str)
        df['type'] = df['type'].fillna('unknown').astype(str).str.lower()
        df['type'] = df['type'].apply(lambda x: x.split(',')[0].strip())
        cleanup_map = {
            "?": "unknown",
            "uncertain": "unknown",
            "call?": "call",
            "song?": "song"
        }
        df['type'] = df['type'].replace(cleanup_map)
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['type'])
        self.num_classes = df['label'].max() + 1
        mapping_dict = {int(index): label for index, label in enumerate(le.classes_)}
        with open('../models/class_mapping.json', 'w') as f:
            json.dump(mapping_dict, f)
        return df.set_index('id')['label']  # id → int label

    def labels_int(self):
        logger.info(f'Loading labels({self.split})')
        ids = [file.stem.split('_')[-3] for file in self.files]  # extract all IDs at once
        labels = self.load_all_metadata().reindex(ids, fill_value=-1).tolist()
        n_missing = labels.count(-1)
        if n_missing > 0:
            logger.warning("%d files have unmatched IDs — will crash during training!", n_missing)
        return labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image), self.labels[idx]
