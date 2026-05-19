import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from ml.datasets.dataset import get_dataloaders
from ml.utils.metrics import calculate_accuracy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

writer = SummaryWriter("ml/experiments/runs")


with open("ml/configs/train_config.yaml", "r") as file:
    config = yaml.safe_load(file)


def validate(model, val_loader, criterion):

    model.eval()

    running_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = calculate_accuracy(all_labels, all_preds)

    return running_loss, accuracy


def train():

    train_loader, val_loader, class_names = get_dataloaders(
        train_dir=config["train_dir"],
        val_dir=config["val_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"]
    )

    model = models.resnet18(weights="DEFAULT")

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    for epoch in range(config["epochs"]):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        writer.add_scalar(
            "Loss/train",
            running_loss,
            epoch
        )

        writer.add_scalar(
            "Loss/val",
            val_loss,
            epoch
        )

        writer.add_scalar(
            "Accuracy/val",
            val_accuracy,
            epoch
        )

        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {running_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

    torch.save(
        model.state_dict(),
        config["save_path"]
    )

    print("\nTraining Complete")


if __name__ == "__main__":
    train()