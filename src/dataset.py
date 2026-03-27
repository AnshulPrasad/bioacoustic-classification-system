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

class BirdSoundDataset():
    def __init__(self, SPLIT_DIR, RAW_DIR, SPECTROGRAM_DIR, transform=None):
        self.SPLIT_DIR = SPLIT_DIR
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

        # drop rare classes before splitting
        KEEP = {'call', 'song', 'alarm call', 'flight call'}
        df = df[df['type'].isin(KEEP)]
        valid_ids = set(df['id'].tolist())

        return df, valid_ids

    def grouped_files(self):
        _, valid_ids = self.id_label()

        # group by audio ids to avoid data leakage
        grouped_files = defaultdict(list)
        for f in self.files:
            rec_id = f.stem.split('_')[-3]
            if rec_id in valid_ids:
                grouped_files[rec_id].append(f)
        return grouped_files

    def stratify(self):
        df, valid_ids = self.id_label()
        grouped_files = self.grouped_files()

        ids_types = df.drop_duplicates('id').set_index('id')['type']
        unique_ids = [i for i in grouped_files.keys() if i in ids_types.index]
        labels_for_unique_ids = [ids_types[i] for i in unique_ids]
        return unique_ids, labels_for_unique_ids

    def split_dataset(self, splits=(0.7, 0.15, 0.15)):
        unique_ids, labels_for_unique_ids = self.stratify()
        grouped_files = self.grouped_files()

        train_ids, temp_ids, _, temp_labels = train_test_split(
            unique_ids, labels_for_unique_ids,
            test_size=1 - splits[0],
            random_state=42,
            stratify=labels_for_unique_ids
        )
        val_ratio = splits[1] / (splits[1] + splits[2])
        val_ids, test_ids, _, _ = train_test_split(
            temp_ids, temp_labels,
            test_size=1 - val_ratio,
            random_state=42,
            stratify=temp_labels
        )

        # copy files
        _paths = []
        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            # freshly remake the folder
            output_path = Path(self.SPLIT_DIR) / split_name
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # iterate over the ids that belong to this split
            paths=[]
            for rec_id in ids:
                for f in grouped_files[rec_id]:
                    file_path = output_path / f.name
                    paths.append(file_path)
                    shutil.copy(f, file_path)
            _paths.append(paths)

        # logger.info("Split complete — train:%d val:%d test:%d recording IDs",
        #             len(train_paths), len(val_paths), len.test_paths))
        return _paths

    def encode(self, paths):
        df, unique_ids = self.id_label()

        # encode labels
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['type'])

        self.num_classes = len(set(df['label']))

        # Save the mapping for Django
        mapping_dict = {int(index): str(label) for index, label in enumerate(le.classes_)}
        with open('../models/class_mapping.json', 'w') as f:
            json.dump(mapping_dict, f)

        ids_labels = df.set_index('id')['label']  # id → int label
        ids=[]
        for path in paths:
            id = path.stem.split('_')[-3]
            ids.append(id)
        list_labels = ids_labels.reindex(ids, fill_value=-1).tolist()
        return list_labels

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
