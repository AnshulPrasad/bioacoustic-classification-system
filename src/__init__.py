import librosa
import yaml
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from download import Species
from preprocess import Preprocessor
from features import FeatureExtractor
from dataset import BirdSoundDataset
from model import Model
from train import Train
from evaluate import Evaluator
from predict import Predictor
from logger import get_logger
logger = get_logger(__name__, 'pipeline.log')

config_path = Path("../configs/config.yaml")  # change path if needed
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
SPECIES_LIST = [(s['scientific_name'], s['common_name']) for s in config['species_list']]
RAW_DIR = config['RAW_DIR']
PROCESSED_DIR = config['PROCESSED_DIR']
SPECTROGRAM_DIR = config["SPECTROGRAM_DIR"]

def download():
    logger.info("Downloading data")
    for sci_name, common_name in SPECIES_LIST:
        try:
            obj = Species(sci_name, RAW_DIR)
            obj.download()
        except ValueError as e:
            logger.warning("Skipping %s: %s", common_name, e)
        except Exception as e:
            logger.error("Unexpected error for %s: %s", common_name, e)

def preprocess():
    logger.info("Preprocessing data")
    for species_folder in sorted(Path(RAW_DIR).iterdir()): # traverse all the files and folders in the directory "data"
        if species_folder.is_dir(): # only use species directories
            for audio_path in Path(species_folder).rglob('*.mp3'): # traverse all the audio files in the sub-folder.
                try:
                    audio, sr = librosa.load(audio_path, sr=22050)
                    obj = Preprocessor(audio, sr)
                    resampled = obj.resample_audio(sr)
                    mono = obj.to_mono(resampled)
                    trimmed = obj.trim_silence(mono)
                    chunks = obj.chunk_audio(trimmed, sr, 5)
                    folder_path = Path(f"{PROCESSED_DIR}/{'_'.join(species_folder.stem.split('_')[:-1])}_wav")
                    folder_path.mkdir(parents=True, exist_ok=True)
                    for i, chunk in enumerate(chunks): # save all the chunks of the audio file in the folder PROCESSED_DIR
                        file_path =  Path(f"{audio_path.stem}_chunk{i}").with_suffix(".wav")
                        output_path = folder_path / file_path
                        if output_path.exists():
                            logger.info("Already exist: %s", output_path)
                            continue
                        obj.save_audio(chunk, output_path, sr=22050)
                        logger.info("Saved chunk: %s", output_path)
                    logger.info("Preprocessed: %s", audio_path)
                except Exception as e:
                    logger.error("Skipping %s: %s", audio_path, e)

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

def split_dataset(species_dir, output_dir, splits=(0.7, 0.15, 0.15)):
    files = list(Path(species_dir).rglob("*.png"))  # mel spectrogram images
    logger.info("Total files in the dataset: %d",len(files))
    train, temp = train_test_split(files, test_size=1-splits[0], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
        output_path = Path(output_dir) / split_name
        if output_path.exists(): # freshly remake the folder
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            file_path = output_path / f.name
            shutil.copy(f, file_path)

def dataset(split):
    obj = BirdSoundDataset('../data/splited', split=split)
    loader = DataLoader(obj, batch_size=32, shuffle=True, num_workers=4)
    return loader, obj

def model(num_classes):
    obj = Model()
    logger.info("Model created")
    return obj.build_model(num_classes)

def train(model, train_loader, val_loader):
    obj = Train(model, train_loader, val_loader)
    obj.train()

def evaluate(test_loader, num_classes):
    obj = Evaluator(test_loader, num_classes)
    obj.evaluate()
    obj.confusion_matrix()
    obj.classify_report()

def predict():
    obj = Predictor()
    obj.to_tensor()
    obj.load()
    obj.evaluate()
    obj.predict()


if __name__ == "__main__":
    download()
    preprocess()
    feature_extraction()
    split_dataset('../data/spectrograms', '../data/splited')
    train_loader, train_dataset = dataset('train')
    val_loader, _ = dataset('val')
    test_loader, _ = dataset('test')
    model = model(train_dataset.num_classes) # num_classes from train dataset
    train(model, train_loader, val_loader)
    evaluate(test_loader, train_dataset.num_classes) # label encoding is same for all sets (train, val, test)