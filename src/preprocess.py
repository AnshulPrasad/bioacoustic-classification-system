# Responsibilities:
# - Resample MP3s to 22050 Hz
# - Convert stereo to mono
# - Trim silence
# - Chunk into 5-second clips
# - Save as WAV to data/processed/

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from logger import get_logger
logger = get_logger(__name__, 'preprocess.log')

class Preprocessor:
    def __init__(self, audio_path: Path):
        self.audio, self.sr = librosa.load(audio_path, sr=22050)

    def resample_audio(self, target_sr = 22050):
        if self.sr == target_sr:
            resampled = self.audio
            return resampled
        resampled = librosa.resample(self.audio, orig_sr=self.sr, target_sr=target_sr)
        return resampled

    @staticmethod
    def to_mono(audio: np.ndarray):
        if audio.ndim == 1:
            return audio
        mono = librosa.to_mono(audio)
        return mono

    @staticmethod
    def trim_silence(audio: np.ndarray):
        audio_trim, _ = librosa.effects.trim(audio)
        return audio_trim

    @staticmethod
    def chunk_audio(audio: np.ndarray, sr: int =22050, duration: int = 5):
        chunk_size = sr * duration
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
        chunks = [librosa.util.fix_length(c, size=chunk_size) for c in chunks]
        return chunks # returns list of chunks

    @staticmethod
    def save_audio(audio: np.ndarray, output_path: Path, sr: int = 22050):
        sf.write(output_path, audio, samplerate=sr)

    def preprocess_audio(self):
        resampled = self.resample_audio(self.sr)
        mono = self.to_mono(resampled)
        trimmed = self.trim_silence(mono)
        chunks = self.chunk_audio(trimmed, self.sr, 5)
        return chunks