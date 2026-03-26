# Responsibilities:
# - Confusion matrix
# - Classification report
# - Per-label accuracy
# - Save plots to outputs/

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from logger import get_logger
logger = get_logger(__name__, 'pipeline.log')

from model import Model

class Evaluator:
    def __init__(self, test_loader, num_classes, MODEL_PATH, CONFUSION_MATRIX_PATH):
        self.test_loader = test_loader
        self.class_names = list(range(num_classes))
        self.all_preds = []
        self.all_labels = []
        obj = Model()
        self.model = obj.build_model(len(self.class_names)) # build a new model
        state = torch.load('../models/checkpoints/best_model.pth')
        self.model.load_state_dict(state)

    def classify_report(self):
        logger.info('Preparing classification report...')
        logger.info(classification_report(self.all_labels, self.all_preds, target_names=self.present_names))
        logger.info('Prepared classification report')

    def confusion_matrix(self):
        logger.info('Preparing confusion matrix...')
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.present_names, yticklabels=self.present_names)
        Path('../outputs').mkdir(parents=True, exist_ok=True)
        plt.savefig('../outputs/confusion_matrix.png')
        logger.info('Prepared confusion matrix')

    def evaluate(self):
        logger.info("Evaluating...")
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                preds = self.model(images).argmax(dim=1)
                self.all_preds.extend(preds.cpu().numpy())
                self.all_labels.extend(labels.numpy())
        # Only use labels that appear in predictions/actuals
        present_labels = sorted(set(self.all_labels) | set(self.all_preds))
        self.present_names = [str(i) for i in present_labels]
        logger.info("Evaluated")
