import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        "ml/saved_models/defect_detector.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        predicted = torch.argmax(outputs, 1)

    return CLASS_NAMES[predicted.item()]


if __name__ == "__main__":

    prediction = predict_image("test.jpg")

    print(f"Prediction: {prediction}")