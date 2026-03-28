# Responsibilities:
# - Generate mel spectrograms from processed WAVs
# - Apply augmentation (noise, pitch shift, time stretch)
# - Save spectrograms as PNGs to data/spectrograms/
# - Split into train/val/test

import librosa
import numpy as np
from PIL import Image
from logger import get_logger
from pathlib import Path
logger = get_logger(__name__, 'features.log')

class FeatureExtractor:
    def __init__(self, audio_path: Path):
        self.audio, self.sr = librosa.load(audio_path, sr=22050)

    def augment_audio(self, sr: int=22050):
        stretched = librosa.effects.time_stretch(self.audio, rate= np.random.uniform(0.8,1.2)) # time stretched
        pitched = librosa.effects.pitch_shift(self.audio, sr=sr, n_steps=np.random.randint(-2,2)) # pitch shift
        noise = np.random.normal(0, 0.005, len(self.audio)) # add background noise
        noisy = self.audio + noise
        return stretched, pitched, noisy

    def generate_melspectrogram(self, audio, sr=22050, n_mels=128, hop_length=512):
    def generate_melspectrogram(audio: np.ndarray, sr:int =22050, n_mels:int =128, hop_length:int =512):
        mel = librosa.feature.melspectrogram(y=audio, n_mels=n_mels, sr=sr, hop_length=hop_length, fmin=500, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max) # convert to decibels
        return mel_db

    def save_spectrogram(self, spectrogram, path, hop_length=512, x_axis="time", y_axis="mel"):
    def save_spectrogram(spectrogram: np.ndarray, path: Path):
        # Normalize to 0-255
        mel_norm = ((spectrogram - spectrogram.min()) /
                    (spectrogram.max() - spectrogram.min()) * 255).astype(np.uint8)
        img = Image.fromarray(mel_norm).convert('RGB')
        img.save(path)