# Responsibilities:
# - Confusion matrix
# - Classification report
# - Per-species accuracy
# - Save plots to outputs/

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class Evaluator:
    def __init__(self, test_loader, num_classes):
        self.test_loader = test_loader
        self.class_names = list(range(num_classes))
        self.all_preds = []
        self.all_labels = []
        obj = Model()
        self.model = obj.build_model(len(self.class_names)) # build a new model
        state = torch.load('../models/checkpoints/best_model.pth')
        self.model.load_state_dict(state)

    def classify_report(self):
        # Classification report
        print(classification_report(self.all_labels, self.all_preds, target_names=self.class_names))

    def confusion_matrix(self):
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.savefig('../outputs/confusion_matrix.png')

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                preds = self.model(images).argmax(dim=1)
                self.all_preds.extend(preds.cpu().numpy())
                self.all_labels.extend(labels.numpy())
