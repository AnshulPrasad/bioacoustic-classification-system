import librosa
from collections import Counter
from torch.utils.data import WeightedRandomSampler
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
from config.config import (RAW_DIR,
                           PROCESSED_DIR,
                           SPECTROGRAM_DIR,
                           SPLIT_DIR,
                           MODEL_PATH,
                           CONFUSION_MATRIX_PATH,
                           SPECIES_LIST,
                           SPLIT_JSON_PATH,
                           )
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
    logger.info("Feature extraction...")

    for processed_folder in PROCESSED_DIR.iterdir():

        # make folder for storing created spectrograms inside SPECTROGRAM_DIR
        folder_path = SPECTROGRAM_DIR / f"{'_'.join(processed_folder.stem.split('_')[:-1])}_png"
        folder_path.mkdir(parents=True, exist_ok=True)

        # loop over all processed .wav files
        for audio_path in processed_folder.glob("*.wav"):

            # feature extraction from an audio file
            audio, sr = librosa.load(audio_path, sr=22050)
            obj = FeatureExtractor(audio, sr)
            stretched, pitched, noisy = obj.augment_audio()

            # store all the versions(audio, stretched, pitched, noisy) of the audio file
            for audio_version, version_name in zip([audio, stretched, pitched, noisy], ['audio', 'stretched', 'pitched', 'noisy']):
                # create mel spectrogram
                mel_db = obj.generate_melspectrogram(audio_version)

                # save
                file_path = f"{audio_path.stem}_{version_name}".with_suffix('.png')
                output_path = folder_path / file_path
                if output_path.exists():
                    logger.info("Already exist: %s", output_path)
                    continue
                else:
                    obj.save_spectrogram(mel_db, output_path)
                    logger.info("Saved: %s", output_path)

def dataset(split):
    obj = BirdSplitDataset(SPLIT_JSON_PATH, split=split)

    # training set must be balanced for avoiding overfitting
    if split == 'train':
        counts = Counter(obj.labels)
        weights = [1.0 / counts[l] for l in obj.labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True) # rebalancing how often each example is drawn
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