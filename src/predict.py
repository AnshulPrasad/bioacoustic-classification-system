# Responsibilities:
# - Load a trained model
# - Accept a raw MP3/WAV file
# - Return predicted species + confidence

import librosa
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from logger import get_logger
logger = get_logger(__name__, 'predict.log')

class Predictor:
    def __init__(self, audio_path, model_path, class_names):
        self.audio_path = audio_path
        self.model_path = model_path
        self.class_names = class_names

    def to_tensor(self, mel_db):
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        img    = Image.fromarray(mel_db).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        return tensor

    def load(self):
        model = self.build_model(len(self.class_names))
        model.load_state_dict(torch.load(self.model_path))
        return model

    def evaluate(self,model, tensor):
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)
            pred = probs.argmax().item()
        return pred, probs

    def predict(self):
        audio, sr = librosa.load(self.audio_path, sr=22050, mono=True)
        mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmin=500, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        tensor = self.to_tensor(mel_db)
        model = self.load()
        pred, probs = self.evaluate(model, tensor)
        logger.info(f"Predicted: {self.class_names[pred]} ({probs[0][pred]:.2%} confidence)")
