import librosa
import numpy as np
from preprocess import Preprocessor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

# Responsibilities:
# - Generate mel spectrograms from processed WAVs
# - Apply augmentation (noise, pitch shift, time stretch)
# - Save spectrograms as PNGs to data/spectrograms/
# - Split into train/val/test

class FeatureExtractor ():

    def __init__(self, audio, original_sr):
        self.audio = audio
        self.original_sr = original_sr

    def augment_audio(self, sr=22050):
        stretched = librosa.effects.time_stretch(self.audio, rate= np.random.uniform(0.8,1.2)) # time stretched
        pitched = librosa.effects.pitch_shift(self.audio, sr=sr, n_steps=np.random.randint(-2,2)) # pitch shift
        noise = np.random.normal(0, 0.005, len(self.audio)) # add background noise
        noisy = self.audio + noise
        return stretched, pitched, noisy

    def generate_melspectrogram(self, audio, sr=22050, n_mels=128, hop_length=512):
        mel = librosa.feature.melspectrogram(y=audio, n_mels=n_mels, sr=sr, hop_length=hop_length, fmin=500, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max) # convert to decibels
        return mel_db

    def save_spectrogram(self, spectrogram, path, hop_length=512, x_axis="time", y_axis="mel"):
        np.save(path.with_suffix('.npy'), spectrogram)
        plt.figure(figsize=(8,3))
        librosa.display.specshow(spectrogram, sr=22050, hop_length=hop_length, x_axis=x_axis, y_axis=y_axis)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()