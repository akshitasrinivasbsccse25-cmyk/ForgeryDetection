"""
model.py
--------
Defines the Hybrid Graph-Frequency Deep Learning model.

Two branches:
  1. CNN branch  -> works on the DCT (frequency domain) image  (TensorFlow/Keras)
  2. GAT branch  -> works on a graph of document fields         (PyTorch Geometric)

Their features are concatenated (fusion) and passed to a Dense classifier
that outputs Genuine / Forged.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# ---------------------------------------------------------------------
# 1. DCT PREPROCESSING
# ---------------------------------------------------------------------
def compute_dct_image(gray_image, size=128):
    """
    Convert a grayscale document image into its frequency-domain (DCT)
    representation. High-frequency noise patterns are a strong signal
    of copy-paste / splicing forgery.
    """
    gray = cv2.resize(gray_image, (size, size))
    gray = np.float32(gray) / 255.0
    dct = cv2.dct(gray)                       # 2D Discrete Cosine Transform
    dct_log = np.log(np.abs(dct) + 1e-6)      # log-scale so it can be visualised/learned
    dct_norm = cv2.normalize(dct_log, None, 0, 1, cv2.NORM_MINMAX)
    return dct_norm.astype(np.float32)


# ---------------------------------------------------------------------
# 2. CNN BRANCH (operates on the DCT image)
# ---------------------------------------------------------------------
def build_cnn_branch(input_shape=(128, 128, 1)):
    """Small CNN that extracts frequency-domain forgery features."""
    inp = layers.Input(shape=input_shape, name="dct_input")
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv1")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3_gradcam")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    feat = layers.Dense(32, activation="relu", name="cnn_features")(x)
    return models.Model(inp, feat, name="CNN_Branch")


# ---------------------------------------------------------------------
# 3. GAT BRANCH (operates on the document-field graph)
# ---------------------------------------------------------------------
FIELD_NAMES = ["Photo", "Name", "DateOfBirth", "Signature", "QRCode", "DocumentNumber"]
NUM_NODES = len(FIELD_NAMES)
NODE_FEAT_DIM = 6  # [x, y, w, h, confidence, present-flag]


class GATBranch(nn.Module):
    """
    Graph Attention Network over the 6 document fields.
    Each node = one field, edges = fully-connected (every field can
    influence every other field, e.g. signature vs name mismatch).
    """
    def __init__(self, in_dim=NODE_FEAT_DIM, hidden=16, out_dim=32, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads)
        self.gat2 = GATConv(hidden * heads, out_dim, heads=1)

    def forward(self, x, edge_index, batch=None):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        # graph-level feature = mean over the 6 field nodes
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        out = []
        for b in batch.unique():
            out.append(x[batch == b].mean(dim=0))
        return torch.stack(out)


def build_fully_connected_edges(num_nodes=NUM_NODES):
    """Every node connected to every other node (undirected)."""
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------
# 4 + 5. FUSION + DENSE CLASSIFIER (fusion done in TensorFlow for simplicity:
# GAT features are computed once in PyTorch and fed in as a plain vector input)
# ---------------------------------------------------------------------
def build_fusion_classifier(cnn_feat_dim=32, gat_feat_dim=32):
    dct_input = layers.Input(shape=(128, 128, 1), name="dct_input")
    gat_input = layers.Input(shape=(gat_feat_dim,), name="gat_features")

    cnn_branch = build_cnn_branch()
    cnn_feat = cnn_branch(dct_input)

    fused = layers.Concatenate(name="fusion_layer")([cnn_feat, gat_input])
    x = layers.Dense(32, activation="relu")(fused)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)  # 1 = Forged, 0 = Genuine

    full_model = models.Model([dct_input, gat_input], out, name="Hybrid_Graph_Frequency_Model")
    return full_model, cnn_branch


# ---------------------------------------------------------------------
# GRAD-CAM (explainability for the CNN branch)
# ---------------------------------------------------------------------
def grad_cam(cnn_branch_submodel, full_model, dct_image, gat_feat_vector, layer_name="conv3_gradcam"):
    """
    Produces a heatmap showing which frequency regions of the DCT image
    influenced the CNN branch's contribution to the final decision.
    """
    conv_layer = cnn_branch_submodel.get_layer(layer_name)
    grad_model = tf.keras.models.Model(
        [cnn_branch_submodel.input], [conv_layer.output, cnn_branch_submodel.output]
    )

    dct_tensor = tf.convert_to_tensor(dct_image.reshape(1, 128, 128, 1))
    gat_tensor = tf.convert_to_tensor(gat_feat_vector.reshape(1, -1))

    with tf.GradientTape() as tape:
        conv_out, cnn_feat = grad_model(dct_tensor)
        tape.watch(conv_out)
        fused = tf.concat([cnn_feat, gat_tensor], axis=1)
        # run the fusion + dense layers on top of the fused vector
        x = full_model.get_layer("dense")(fused) if full_model.get_layer("dense") else fused
        pred = full_model.get_layer("output")(x)

    grads = tape.gradient(pred, conv_out)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = cv2.resize(heatmap, (128, 128))
    return heatmap
