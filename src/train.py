# Responsibilities:
# - Training loop
# - Validation loop
# - Early stopping
# - Save best model

import torch
import torch.nn as nn
from logger import get_logger
logger = get_logger(__name__, 'train.log')

class Train:
    def __init__(self, model, train_loader, val_loader, MODEL_PATH, epochs=50, lr=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.MODEL_PATH = MODEL_PATH
        self.epochs = epochs
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0

    def train_one_epoch(self, epoch):
        logger.info("Training epoch: %d", epoch)
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

    def validate_one_epoch(self, epoch):
        logger.info("Validating epoch: %d", epoch)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        logger.info("Epoch %d | Val Acc: %.4f", epoch + 1, val_acc)
        logger.info("Validated epoch: %d", epoch)
        return val_acc

    def save_best_model(self, val_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(self.model.state_dict(), f"{self.MODEL_PATH}")
            logger.info("New best model saved | Val Acc: %.4f", val_acc)

    def train(self):
        logger.info("Training...")
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch) # Training
            val_acc = self.validate_one_epoch(epoch) # Validation
            self.save_best_model(val_acc) # Save best model
        logger.info("Trained")