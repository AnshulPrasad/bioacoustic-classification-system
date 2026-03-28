# Responsibilities:
# - Confusion matrix
# - Classification report
# - Per-label accuracy
# - Save plots to outputs/

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from logger import get_logger
logger = get_logger(__name__, 'evaluate.log')

from model import Model

class Evaluator:
    def __init__(self, test_loader: DataLoader, num_classes: int, model_path: Path, confusion_matrix_path: Path):
        self.test_loader = test_loader
        self.class_names = list(range(num_classes))
        self.MODEL_PATH = model_path
        self.CONFUSION_MATRIX_PATH = confusion_matrix_path

        self.all_preds = []
        self.all_labels = []

        obj = Model()
        self.model = obj.build_model(len(self.class_names)) # build a new model

        state = torch.load(self.MODEL_PATH, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def classify_report(self):
        logger.info('Preparing classification report...')
        logger.info(classification_report(self.all_labels, self.all_preds, target_names=self.present_names))

    def _confusion_matrix(self):
        logger.info('Preparing confusion matrix...')
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.present_names, yticklabels=self.present_names)
        plt.savefig(self.CONFUSION_MATRIX_PATH)

    def evaluate(self):
        logger.info("Evaluating...")
        self.model.eval()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                preds = self.model(images).argmax(dim=1)

                self.all_preds.extend(preds.cpu().numpy())
                self.all_labels.extend(labels.numpy())

        # Only use labels that appear in predictions/actuals
        present_labels = sorted(set(self.all_labels) | set(self.all_preds))
        self.present_names = [str(i) for i in present_labels]
