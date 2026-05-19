import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

from ml.datasets.dataset import get_dataloaders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():

    train_loader, class_names = get_dataloaders(
        data_dir="data/raw",
        batch_size=16
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
        lr=0.001
    )

    EPOCHS = 5

    for epoch in range(EPOCHS):

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

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Loss: {running_loss:.4f}")

    torch.save(
        model.state_dict(),
        "ml/saved_models/defect_detector.pth"
    )

    print("Model Saved Successfully")


if __name__ == "__main__":
    train()