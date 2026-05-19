import os
import random
import shutil


SOURCE_DIR = "data/raw/NEU-DET"
TRAIN_DIR = "data/raw/train"
VAL_DIR = "data/raw/val"

SPLIT_RATIO = 0.8


def create_dirs(classes):

    for cls in classes:

        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)


def split_data():

    classes = os.listdir(SOURCE_DIR)

    create_dirs(classes)

    for cls in classes:

        class_path = os.path.join(SOURCE_DIR, cls)

        images = os.listdir(class_path)

        random.shuffle(images)

        split_index = int(len(images) * SPLIT_RATIO)

        train_images = images[:split_index]
        val_images = images[split_index:]

        for img in train_images:

            src = os.path.join(class_path, img)
            dst = os.path.join(TRAIN_DIR, cls, img)

            shutil.copy(src, dst)

        for img in val_images:

            src = os.path.join(class_path, img)
            dst = os.path.join(VAL_DIR, cls, img)

            shutil.copy(src, dst)

    print("Dataset split completed")


if __name__ == "__main__":
    split_data()