import logging
import librosa
import os
from pathlib import Path

from preprocess import Preprocessor
from download import Species
from configs.config import SPECIES_LIST
from features import FeatureExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '../data/raw')
LOG_DIR = os.path.dirname(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(os.path.join(LOG_DIR, '../logs/download.log'), encoding='utf-8', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s :: %(message)s'))
logger.addHandler(file_handler)


for sci_name, common_name in SPECIES_LIST:
    try:
        Species(sci_name).download()
    except ValueError as e:
        logger.warning("Skipping %s: %s", common_name, e)
    except Exception as e:
        logger.error("Unexpected error for %s: %s", common_name, e)


for species_folder in Path('data/raw').rglob('*'):
    for audio_path in Path(species_folder).rglob('*.mp3'):
        audio, sr = librosa.load(audio_path, sr=22050)
        obj = Preprocessor(audio, sr)
        resampled = obj.resample_audio(sr)
        monoed = obj.to_mono(resampled)
        trimed = obj.trim_silence(monoed)
        chunked = obj.chunk_audio(trimed, sr, 5)

        output_path = Path('data/processed') / audio_path.stem + '.wav'
        obj.save_audio(audio, output_path, sr=22050)
        del obj


for processed_species_folder in Path('data/processed').rglob('*'):
    for processed_audio_path in Path(processed_species_folder).rglob('*.wav'):
        audio, sr = librosa.load(processed_audio_path, sr=22050)
        obj = FeatureExtractor(audio, sr)
        mel_db = obj.generate_melspectrogram(sr)
        obj.augment_audio(mel_db)

        output_path = Path('data/spectrograms') / processed_audio_path.stem + '.png'
        obj.save_spectrogram(mel_db, output_path)

        obj.split_dataset('data/processed', 'data/spectrograms')
        del obj