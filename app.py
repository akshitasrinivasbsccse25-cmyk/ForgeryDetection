"""
app.py
------
Flask backend for the Hybrid Graph-Frequency Identity Document Forgery
Detection system. Loads the trained CNN+Fusion (Keras .h5) and GAT
(PyTorch .pth) weights ONCE at startup, then performs real inference
(no random / hardcoded outputs) on every uploaded document image.
"""

import os
import time
import numpy as np
import cv2
import torch
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from model import compute_dct_image, build_fusion_classifier, GATBranch, grad_cam
from utils import build_graph_features, load_image, to_gray, overlay_heatmap_on_image, image_to_base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "models/cnn_gat_fusion.h5"
GAT_PATH = "models/gat_branch.pth"
GAT_FEAT_DIM = 32

# ---------------------------------------------------------------------
# LOAD TRAINED MODELS AT STARTUP (real weights, no random predictions)
# ---------------------------------------------------------------------
gat_branch = GATBranch()
full_model = None
cnn_branch = None
MODEL_READY = False

try:
    gat_branch.load_state_dict(torch.load(GAT_PATH, map_location="cpu"))
    gat_branch.eval()

    full_model, cnn_branch = build_fusion_classifier(gat_feat_dim=GAT_FEAT_DIM)
    full_model.load_weights(MODEL_PATH)
    MODEL_READY = True
    print("Loaded trained model weights successfully.")
except Exception as e:
    print("WARNING: trained weights not found (%s). "
          "Run train.py first to train and save models/. "
          "Server will still start but /predict will return an error "
          "until weights exist." % e)


# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Save the uploaded document image and return its path for preview."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    return jsonify({"filepath": save_path})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Runs the REAL hybrid inference pipeline:
      1. DCT transform of uploaded image        -> CNN branch input
      2. OCR field detection -> graph            -> GAT branch input
      3. Feature fusion + Dense classifier        -> prediction
      4. Grad-CAM heatmap of the CNN branch       -> explainability
    """
    if not MODEL_READY:
        return jsonify({"error": "Model weights not found. Please run train.py first."}), 500

    data = request.get_json()
    filepath = data.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Invalid file path"}), 400

    start_time = time.time()

    # ---- Step 1: DCT (frequency domain) preprocessing ----
    img = load_image(filepath)
    gray = to_gray(img)
    dct_img = compute_dct_image(gray, size=128)
    dct_input = dct_img.reshape(1, 128, 128, 1)

    # ---- Step 2: OCR-based document field graph + GAT branch ----
    x_nodes, edge_index, _ = build_graph_features(img)
    with torch.no_grad():
        gat_feat = gat_branch(x_nodes, edge_index).numpy()  # shape (1, 32)

    # ---- Step 3: Fusion + Dense classifier (REAL forward pass) ----
    prob_forged = float(full_model.predict([dct_input, gat_feat], verbose=0)[0][0])

    if prob_forged >= 0.5:
        prediction = "Forged"
        confidence = prob_forged * 100
    else:
        prediction = "Genuine"
        confidence = (1 - prob_forged) * 100

    # ---- Step 4: Grad-CAM explainability ----
    try:
        heatmap = grad_cam(cnn_branch, full_model, dct_img, gat_feat[0])
        overlay = overlay_heatmap_on_image(img, heatmap)
        heatmap_b64 = image_to_base64(overlay)
    except Exception as e:
        print("Grad-CAM failed:", e)
        heatmap_b64 = ""

    elapsed = time.time() - start_time

    return jsonify({
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%",
        "processing_time": f"{elapsed:.2f}s",
        "heatmap": heatmap_b64
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
