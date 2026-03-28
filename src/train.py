# Responsibilities:
# - Training loop
# - Validation loop
# - Early stopping
# - Save best model

import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from logger import get_logger
from model import Model
logger = get_logger(__name__, 'train.log')

class Train:
    def __init__(self, model: Model, train_loader:DataLoader, val_loader:DataLoader, model_path: Path, epochs: int=50, lr: float=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.MODEL_PATH = MODEL_PATH
        self.epochs = epochs
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0

    def train_one_epoch(self, epoch: int):
        logger.info("Training epoch: %d/%d", epoch, self.epochs)
        self.model.train()
        total_loss=0.0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        logger.info("Epoch %d | Train Loss: %.4f", epoch, avg_loss)

    def validate_one_epoch(self, epoch: int):
        logger.info("Validating epoch: %d", epoch)
        self.model.eval()
        correct, total = 0, 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for pred, label in zip(preds.cpu(), labels.cpu()):
                    class_total[label.item()] += 1
                    if pred == label:
                        class_correct[label.item()] += 1
        val_acc = correct / total
        logger.info("Epoch %d | Val Acc: %.4f", epoch, val_acc)

        for cls in sorted(class_total):
            acc = class_correct[cls] / class_total[cls]
            logger.info("Class %d | Acc: %.4f (%d samples)", cls, acc, class_total[cls])

        return val_acc

    def save_best_model(self, val_acc: float):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(self.model.state_dict(), f"{self.MODEL_PATH}")
            logger.info("New best model saved | Val Acc: %.4f", val_acc)

    def train(self):
        logger.info("Training...")
        for epoch in range(1, 1+self.epochs):
            self.train_one_epoch(epoch) # Training
            val_acc = self.validate_one_epoch(epoch) # Validation
            self.scheduler.step(val_acc)
            self.save_best_model(val_acc) # Save best model
        logger.info("Trained")