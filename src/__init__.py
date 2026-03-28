import librosa
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from download import Species
from preprocess import Preprocessor
from features import FeatureExtractor
from dataset import BirdSoundDataset, BirdSplitDataset
from model import Model
from train import Train
from evaluate import Evaluator
from predict import Predictor
from logger import get_logger
from config.config import RAW_DIR, PROCESSED_DIR, SPECTROGRAM_DIR, MODEL_PATH, CONFUSION_MATRIX_PATH, SPECIES_LIST
logger = get_logger(__name__, '__init__.log')

def download():
    logger.info("Downloading data...")
    for sci_name, common_name in SPECIES_LIST: # remove common_name: no use
        obj = Species(sci_name, RAW_DIR)
        obj.download()

def preprocess():
    logger.info("Preprocessing data...")

    # loop over files (.csv) + folders (species) in the directory data/raw
    for raw_folder in RAW_DIR.iterdir():

        # pick only folders
        if raw_folder.is_dir():

            # make folder for storing processed files inside PROCESSED_DIR
            species_folder_path = f"{'_'.join(raw_folder.stem.split('_')[:-1])}_wav"
            folder_path = PROCESSED_DIR / species_folder_path
            folder_path.mkdir(parents=True, exist_ok=True)

            # loop over the audio(.mp3) files
            for audio_path in raw_folder.rglob('*.mp3'):

                # preprocess an audio file
                audio, sr = librosa.load(audio_path, sr=22050)
                obj = Preprocessor(audio, sr)
                resampled = obj.resample_audio(sr)
                mono = obj.to_mono(resampled)
                trimmed = obj.trim_silence(mono)
                chunks = obj.chunk_audio(trimmed, sr, 5)


                # save the chunks of the audio file in the folder (if not saved)
                for i, chunk in enumerate(chunks):
                    file_path =  f"{audio_path.stem}_chunk{i}".with_suffix(".wav")
                    output_path = folder_path / file_path
                    if output_path.exists():
                        logger.info("Already exist: %s", output_path)
                        continue
                    else:
                        obj.save_audio(chunk, output_path, sr=22050)
                        logger.info("Saved chunk: %s", output_path)

def feature_extraction():
    logger.info("Feature extraction")
    for processed_audio_folder in sorted(Path(PROCESSED_DIR).iterdir()):
        for audio_path in Path(processed_audio_folder).glob('*.wav'):
            try:
                audio, sr = librosa.load(audio_path, sr=22050)
                obj = FeatureExtractor(audio, sr)
                stretched, pitched, noisy = obj.augment_audio()
                folder_path = Path(f"{SPECTROGRAM_DIR}/{'_'.join(audio_path.stem.split('_')[:-2])}_png")
                folder_path.mkdir(parents=True, exist_ok=True)
                for audio_version, version_name in zip([audio, stretched, pitched, noisy], ['audio', 'stretched', 'pitched', 'noisy']):
                    mel_db = obj.generate_melspectrogram(audio_version)
                    file_path = Path(f"{audio_path.stem}_{version_name}").with_suffix('.png')
                    output_path = folder_path / file_path
                    if output_path.exists():
                        logger.info("Already exist: %s", output_path)
                        continue
                    obj.save_spectrogram(mel_db, output_path)
                    logger.info("Saved: %s", output_path)
                logger.info("Extracted: %s", audio_path)
            except Exception as e:
                logger.error("Skipping %s: %s", audio_path, e)

def dataset(split):
    obj = BirdSplitDataset("../models/split_index.json", split=split)
    if split == 'train':
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        counts = Counter(obj.labels)
        weights = [1.0 / counts[l] for l in obj.labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(obj, batch_size=32, sampler=sampler, num_workers=4)
    else:
        loader = DataLoader(obj, batch_size=32, shuffle=False, num_workers=4)
    return loader, obj

def model(num_classes):
    obj = Model()
    _model = obj.build_model(num_classes)
    logger.info("Model created")
    return _model

def train(model, train_loader, val_loader):
    obj = Train(model, train_loader, val_loader, MODEL_PATH)
    obj.train()

def evaluate(test_loader, num_classes):
    obj = Evaluator(test_loader, num_classes, MODEL_PATH, CONFUSION_MATRIX_PATH)
    obj.evaluate()
    obj._confusion_matrix()
    obj.classify_report()


if __name__ == "__main__":
    download()
    preprocess()
    feature_extraction()
    builder = BirdSoundDataset(SPLIT_DIR, RAW_DIR, SPECTROGRAM_DIR)
    builder.build_and_save_index("../models/split_index.json")
    train_loader, train_dataset = dataset('train')
    val_loader, _ = dataset('val')
    test_loader, _ = dataset('test')
    m = model(train_dataset.num_classes) # num_classes from train dataset
    train(m, train_loader, val_loader)
    evaluate(test_loader, train_dataset.num_classes) # label encoding is same for all sets (train, val, test)