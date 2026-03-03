import torch
from src.utils import device


class BaseTrainer:
    """Trainer for the custom TransformerClassifier (TensorDataset-based loaders)."""

    def __init__(self, model, criterion, optimizer, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def fit(self, num_epochs: int):
        self.num_batches = len(self.train_loader)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate_one_epoch()
            print(
                f"{self.num_batches}/{self.num_batches} - "
                f"train_loss: {train_loss:.4f} - train_accuracy: {train_acc * 100:.2f}% - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc * 100:.2f}%"
            )

    def train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss / self.num_batches, correct / total

    def evaluate(self, loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / len(loader), correct / total

    def validate_one_epoch(self):
        return self.evaluate(self.val_loader)


class FineTunedBaseTrainer:
    """Trainer for BERT-based models (HuggingFace Dataset-based loaders)."""

    def __init__(self, model, criterion, optimizer, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def fit(self, num_epochs: int):
        self.num_batches = len(self.train_loader)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate_one_epoch()
            print(
                f"{self.num_batches}/{self.num_batches} - "
                f"train_loss: {train_loss:.4f} - train_accuracy: {train_acc * 100:.2f}% - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc * 100:.2f}%"
            )

    def train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss / self.num_batches, correct / total

    def evaluate(self, loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / len(loader), correct / total

    def validate_one_epoch(self):
        return self.evaluate(self.val_loader)
