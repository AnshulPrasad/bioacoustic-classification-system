# Responsibilities:
# - PyTorch Dataset class for loading spectrograms
# - DataLoaders for train/val/test
# - Label encoding

import json
import shutil
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from logger import get_logger
logger = get_logger(__name__, 'dataset.log')

class BirdSoundDataset:
    def __init__(self, split_dir: Path, raw_dir: Path, spectrogram_dir: Path, class_mapping_json: Path):
        self.SPLIT_DIR = split_dir
        self.RAW_DIR = raw_dir
        self.SPECTROGRAM_DIR = spectrogram_dir
        self.CLASS_MAPPING_JSON = class_mapping_json

        self.files = list(Path(self.SPECTROGRAM_DIR).rglob("*.png"))

        self.df = self.data_frame()
        self.valid_ids = set(self.df['id'].tolist())
        self.grouped_files_list = self.grouped_files()
        self.ids, self.labels = self.ids_and_types()

        # encode labels
        le = LabelEncoder()
        self.df['label'] = le.fit_transform(self.df['type'])
        self.num_classes = len(set(self.df['label']))

        # Save the mapping for Django
        mapping_dict = {int(index): str(label) for index, label in enumerate(le.classes_)}
        with open(self.CLASS_MAPPING_JSON, 'w') as f:
            json.dump(mapping_dict, f)

        self.train_paths, self.val_paths, self.test_paths = self.split_dataset()

        self.train_labels = self.encode(self.train_paths)
        self.val_labels = self.encode(self.val_paths)
        self.test_labels = self.encode(self.test_paths)

    def build_and_save_index(self, split_json_path: Path):
        data = {
            "train_paths": [str(p) for p in self.train_paths],
            "val_paths": [str(p) for p in self.val_paths],
            "test_paths": [str(p) for p in self.test_paths],
            "train_labels": self.train_labels,
            "val_labels": self.val_labels,
            "test_labels": self.test_labels,
            "num_classes": int(self.num_classes),
        }
        with open(split_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def data_frame(self):
        # concatenate all csvs and convert ids from int to str
        dfs = [pd.read_csv(f, usecols=['id', 'type']) for f in sorted(Path(self.RAW_DIR).glob("*.csv"))]
        df = pd.concat(dfs,ignore_index=True)
        df['id'] = df['id'].astype(str)

        # clean labels
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
        keep = {"call", "song", "alarm call", "flight call"}
        df = df[df["type"].isin(keep)]

        return df

    def grouped_files(self):
        # group by audio ids to avoid data leakage
        grouped_files = defaultdict(list)
        for f in self.files:
            rec_id = f.stem.split('_')[-3]
            if rec_id in self.valid_ids:
                grouped_files[rec_id].append(f)
        return grouped_files

    def ids_and_types(self):
        ids_types = self.df.drop_duplicates("id").set_index("id")["type"]
        unique_ids = [i for i in self.grouped_files_list.keys() if i in ids_types.index]
        labels_for_unique_ids = [ids_types[i] for i in unique_ids]
        return unique_ids, labels_for_unique_ids

    def split_dataset(self, splits=(0.7, 0.15, 0.15)):
        train_ids, temp_ids, _, temp_labels = train_test_split(
            self.ids, self.labels,
            test_size=1 - splits[0],
            random_state=42,
            stratify=self.labels
        )

        val_ratio = splits[1] / (splits[1] + splits[2])
        val_ids, test_ids, _, _ = train_test_split(
            temp_ids, temp_labels,
            test_size=1 - val_ratio,
            random_state=42,
            stratify=temp_labels
        )

        # copy files
        paths_list = []
        for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            # freshly remake the folder
            output_path = self.SPLIT_DIR / split_name
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # iterate over the ids that belong to this split
            paths=[]
            for rec_id in ids:
                for f in self.grouped_files_list[rec_id]:
                    file_path = output_path / f.name
                    paths.append(file_path)
                    shutil.copy(f, file_path)
            paths_list.append(paths)

        logger.info("Split complete — train:%d val:%d test:%d recording IDs",
                    len(paths_list[0]), len(paths_list[1]), len(paths_list[2]))
        return paths_list

    def encode(self, paths):
        ids_labels = self.df.set_index('id')['label']  # id → int label
        ids=[]
        for path in paths:
            id = path.stem.split('_')[-3]
            ids.append(id)
        labels = ids_labels.reindex(ids, fill_value=-1).tolist()
        return labels

class BirdSplitDataset(Dataset):
    def __init__(self, split_json_path: Path, split: str="train", transform=None):
        with open(split_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if split == "train":
            self.paths = data["train_paths"]
            self.labels = data["train_labels"]
            logger.info("Train: %d %d",len(self.paths), len(self.labels))
        elif split == "val":
            self.paths = data["val_paths"]
            self.labels = data["val_labels"]
            logger.info("Val: %d %d", len(self.paths), len(self.labels))
        elif split == "test":
            self.paths = data["test_paths"]
            self.labels = data["test_labels"]
            logger.info("Test: %d %d", len(self.paths), len(self.labels))
        else:
            raise ValueError("split must be train/val/test")

        self.num_classes = data["num_classes"]

        # Clean, non-random transforms for Val and Test sets!
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]