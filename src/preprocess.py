import librosa
import soundfile as sf
import numpy as np

# Responsibilities:
# - Resample MP3s to 22050 Hz
# - Convert stereo to mono
# - Trim silence
# - Chunk into 5-second clips
# - Save as WAV to data/processed/

class Preprocessor:
    def __init__(self, audio, original_sr):
        self.audio = audio
        self.orig_sr = original_sr

    def resample_audio(self, target_sr=22050):
        if self.orig_sr == target_sr:
            resampled = self.audio
            return resampled
        resampled = librosa.resample(self.audio, orig_sr=self.orig_sr, target_sr=target_sr)
        return resampled

    def to_mono(self, audio):
        if audio.ndim == 1:
            return audio
        monoed = librosa.to_mono(audio)
        return monoed

    def trim_silence(self, audio):
        audio_trim, _ = librosa.effects.trim(audio)
        return audio_trim

    def chunk_audio(self, audio, sr=22050, duration=5):
        chunk_size = sr * duration
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
        chunks = [librosa.util.fix_length(c, size=chunk_size) for c in chunks]
        return chunks # returns list of chunks

    def save_audio(self, audio, output_path, sr=22050):
        sf.write(output_path, audio, samplerate=sr)
