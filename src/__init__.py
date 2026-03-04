import librosa
import yaml
import shutil
from pathlib import Path
from preprocess import Preprocessor
from download import Species
from features import FeatureExtractor
from sklearn.model_selection import train_test_split
from logger import get_logger
logger = get_logger(__name__, 'pipeline.log')


config_path = Path("../configs/config.yaml")  # change path if needed
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
SPECIES_LIST = [(s['scientific_name'], s['common_name']) for s in config['species_list']]

def download():
    for sci_name, common_name in SPECIES_LIST:
        try:
            Species(sci_name).download()
        except ValueError as e:
            logger.warning("Skipping %s: %s", common_name, e)
        except Exception as e:
            logger.error("Unexpected error for %s: %s", common_name, e)

def preprocess():
    logger.info("Preprocessing data")
    for species_folder in Path('../data/raw').rglob('*'):
        for audio_path in Path(species_folder).rglob('*.mp3'):
            try:
                audio, sr = librosa.load(audio_path, sr=22050)
                obj = Preprocessor(audio, sr)
                resampled = obj.resample_audio(sr)
                monoed = obj.to_mono(resampled)
                trimed = obj.trim_silence(monoed)
                chunks = obj.chunk_audio(trimed, sr, 5)
                for i, chunk in enumerate(chunks):
                    output_path = (Path('../data/processed') / f"{audio_path.stem}_chunk{i}").with_suffix(".wav")
                    obj.save_audio(chunk, output_path, sr=22050)
                    logger.info("Saved %s", output_path)
                del obj
                logger.info("Prerocessed %s", audio_path)
            except Exception as e:
                logger.error("Skipping %s: %s", audio_path, e)

def feature_extraction():
    logger.info("Feature extraction")
    for processed_audio_path in Path('../data/processed').rglob('*.wav'):
        # logger.info('Processing %s', processed_audio_path)
        try:
            audio, sr = librosa.load(processed_audio_path, sr=22050)
            obj = FeatureExtractor(audio, sr)
            stretched, pitched, noisy = obj.augment_audio()
            for audio_version, version_name in zip([audio, stretched, pitched, noisy], ['_audio', '_stretched', '_pitched', '_noisy']):
                mel_db = obj.generate_melspectrogram(audio_version)
                output_path = (Path('../data/spectrograms/all') / f"{processed_audio_path.stem}{version_name}").with_suffix(f".png")
                obj.save_spectrogram(mel_db, output_path)
                logger.info("Saved %s", output_path)

            del obj
            logger.info("Extracted %s", processed_audio_path)
        except Exception as e:
            logger.error("Skipping %s: %s", processed_audio_path, e)

def split_dataset(species_dir, output_dir, splits=(0.7, 0.15, 0.15)):
    files = list(Path(species_dir).rglob("*.png"))  # mel spectrogram images
    logger.info(len(files))
    train, temp = train_test_split(files, test_size=1 - splits[0], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
        for f in split_files:
            dest = Path(output_dir) / split_name / f.name
            dest.touch()
            shutil.copy(f, dest)

if __name__ == "__main__":
    download()
    preprocess()
    feature_extraction()
    split_dataset('../data/spectrograms/all', '../data/spectrograms')
