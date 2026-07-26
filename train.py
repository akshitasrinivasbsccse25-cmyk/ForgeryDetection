"""
train.py
--------
Trains the Hybrid Graph-Frequency model.

Expected dataset layout (download from Kaggle, e.g. search:
"Passport Forgery Dataset", "PAN Card Fraud", "Aadhaar Card Dataset",
"Driving Licence Dataset" and merge them into this structure):

    dataset/
        train/
            genuine/   *.jpg
            forged/    *.jpg
        test/
            genuine/   *.jpg
            forged/    *.jpg

Run:
    python train.py
Produces:
    models/cnn_gat_fusion.h5   (Keras weights for CNN + fusion + dense head)
    models/gat_branch.pth      (PyTorch GAT branch weights)
"""

import os
import glob
import numpy as np
import cv2
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import compute_dct_image, build_fusion_classifier, GATBranch
from utils import build_graph_features, load_image, to_gray

DATASET_DIR = "dataset"
IMG_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 8


def gather_filepaths(split):
    genuine = glob.glob(os.path.join(DATASET_DIR, split, "genuine", "*"))
    forged = glob.glob(os.path.join(DATASET_DIR, split, "forged", "*"))
    paths = genuine + forged
    labels = [0] * len(genuine) + [1] * len(forged)  # 0=Genuine, 1=Forged
    return paths, labels


def augment_image(gray_img):
    """Simple augmentation: random flip + brightness jitter (kept lightweight)."""
    if np.random.rand() > 0.5:
        gray_img = cv2.flip(gray_img, 1)
    factor = 0.8 + np.random.rand() * 0.4
    gray_img = np.clip(gray_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return gray_img


def build_dataset_arrays(paths, labels, gat_branch, augment=False):
    """
    For every image: compute the DCT image (CNN input) and the GAT
    graph-level feature vector (using the untrained/frozen GAT branch
    as a feature extractor over OCR-derived node positions).
    """
    dct_batch, gat_batch, y = [], [], []

    for path, label in zip(paths, labels):
        img = load_image(path)
        gray = to_gray(img)
        if augment:
            gray = augment_image(gray)

        dct_img = compute_dct_image(gray, size=IMG_SIZE)
        dct_batch.append(dct_img[..., np.newaxis])

        x_nodes, edge_index, _ = build_graph_features(img)
        with torch.no_grad():
            gat_feat = gat_branch(x_nodes, edge_index).numpy().flatten()
        gat_batch.append(gat_feat)

        y.append(label)

    return (np.array(dct_batch, dtype=np.float32),
            np.array(gat_batch, dtype=np.float32),
            np.array(y, dtype=np.float32))


def main():
    os.makedirs("models", exist_ok=True)

    train_paths, train_labels = gather_filepaths("train")
    test_paths, test_labels = gather_filepaths("test")

    if len(train_paths) == 0:
        print("No training images found in dataset/train/{genuine,forged}. "
              "Download Kaggle passport/PAN/Aadhaar/driving-licence datasets "
              "and place images accordingly before running train.py.")
        return

    # GAT branch is used here purely as a structural feature extractor.
    # For a full end-to-end paper implementation you would jointly train
    # the GAT + CNN together with a custom PyTorch training loop; here we
    # keep it simple and beginner-friendly by pre-computing GAT features.
    gat_branch = GATBranch()
    gat_branch.eval()
    torch.save(gat_branch.state_dict(), "models/gat_branch.pth")

    print("Building training arrays (DCT + GAT features)...")
    X_dct_train, X_gat_train, y_train = build_dataset_arrays(
        train_paths, train_labels, gat_branch, augment=True)
    X_dct_test, X_gat_test, y_test = build_dataset_arrays(
        test_paths, test_labels, gat_branch, augment=False)

    # Normalize DCT images already done in compute_dct_image (0-1 range).
    full_model, cnn_branch = build_fusion_classifier(gat_feat_dim=X_gat_train.shape[1])
    full_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    full_model.summary()

    full_model.fit(
        [X_dct_train, X_gat_train], y_train,
        validation_data=([X_dct_test, X_gat_test], y_test) if len(test_paths) else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    full_model.save("models/cnn_gat_fusion.h5")
    print("Training complete. Saved models/cnn_gat_fusion.h5 and models/gat_branch.pth")


if __name__ == "__main__":
    main()
