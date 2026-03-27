# Responsibilities:
# - Load a trained model
# - Accept a raw MP3/WAV file
# - Return predicted species + confidence

import json
import torch
import librosa
import numpy as np
from torchvision import transforms
from PIL import Image
from model import Model
from logger import get_logger
logger = get_logger(__name__, 'predict.log')

class Predictor:
    def __init__(self, model_path="../models/checkpoints/best_model.pth", mapping_path="../models/class_mapping.json"):
        self.model_path = model_path
        self.mapping_path = mapping_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Load the JSON label mapping
        with open(self.mapping_path, 'r') as f:
            self.class_mapping = {int(k): v for k, v in json.load(f).items()}

        self.num_classes = len(self.class_mapping)

        # 2. Match the training Dataset transforms EXACTLY
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. Load the model into memory ONCE when the Django server starts
        self.model = self.load_model()

    def load_model(self):
        logger.info("Loading model into memory...")
        obj = Model()
        model = obj.build_model(self.num_classes)
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        logger.info("Model loaded successfully.")
        return model

    def process_audio(self, audio_path):
        logger.info(f"Processing audio: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)

        # Pad or trim to exactly 5 seconds so the tensor matches the trained model
        audio = librosa.util.fix_length(audio, size=22050 * 5)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmin=500, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        img = Image.fromarray(mel_db).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, audio_path):
        logger.info("Running prediction...")
        tensor = self.process_audio(audio_path)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        pred_class = int(pred_idx.item())
        confidence_score = float(confidence.item())
        predicted_label = self.class_mapping.get(pred_class, "Unknown")

        logger.info(f"Predicted: {predicted_label} ({confidence_score:.2%} confidence)")

        return predicted_label, confidence_score