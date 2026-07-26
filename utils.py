"""
utils.py
--------
Helper functions:
  - Reading images
  - Running OCR/layout detection to find document field boxes
  - Turning field boxes into a graph (node features + edge index) for the GAT branch
  - Converting Grad-CAM heatmap into a displayable overlay image
"""

import cv2
import numpy as np
import base64
import torch

from model import FIELD_NAMES, NUM_NODES, NODE_FEAT_DIM, build_fully_connected_edges

# EasyOCR is loaded lazily (it is slow to import) and only once.
_reader = None


def get_ocr_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not read image at " + path)
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------
# OCR / LAYOUT DETECTION -> DOCUMENT FIELD GRAPH NODES
# ---------------------------------------------------------------------
def detect_fields(img):
    """
    Uses EasyOCR to find text regions on the document, then heuristically
    assigns each detected box to one of the 6 known document fields based
    on simple keyword / position rules. Falls back to an approximate
    layout (typical ID-card positions) if OCR finds nothing, so the
    pipeline always produces 6 graph nodes.
    """
    h, w = img.shape[:2]
    boxes = {}

    try:
        reader = get_ocr_reader()
        results = reader.readtext(img)
    except Exception:
        results = []

    for (bbox, text, conf) in results:
        text_l = text.lower()
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = min(xs), min(ys)
        bw, bh = max(xs) - x, max(ys) - y

        if any(k in text_l for k in ["dob", "birth", "date"]):
            boxes["DateOfBirth"] = (x, y, bw, bh, conf)
        elif any(k in text_l for k in ["name"]):
            boxes["Name"] = (x, y, bw, bh, conf)
        elif any(k in text_l for k in ["signature", "sign"]):
            boxes["Signature"] = (x, y, bw, bh, conf)
        elif any(k in text_l for k in ["no", "number", "id"]) and "DocumentNumber" not in boxes:
            boxes["DocumentNumber"] = (x, y, bw, bh, conf)

    # Photo & QR code are visual (not text), approximate typical ID-card layout
    boxes.setdefault("Photo", (0.05 * w, 0.15 * h, 0.25 * w, 0.35 * h, 0.5))
    boxes.setdefault("QRCode", (0.75 * w, 0.60 * h, 0.20 * w, 0.20 * h, 0.5))
    boxes.setdefault("Name", (0.35 * w, 0.20 * h, 0.4 * w, 0.08 * h, 0.3))
    boxes.setdefault("DateOfBirth", (0.35 * w, 0.35 * h, 0.3 * w, 0.08 * h, 0.3))
    boxes.setdefault("Signature", (0.35 * w, 0.75 * h, 0.3 * w, 0.1 * h, 0.3))
    boxes.setdefault("DocumentNumber", (0.05 * w, 0.55 * h, 0.4 * w, 0.08 * h, 0.3))

    return boxes, (h, w)


def build_graph_features(img):
    """
    Builds the node-feature matrix (6 x NODE_FEAT_DIM) and the fully
    connected edge index used by the GAT branch.
    Node feature = [x_norm, y_norm, w_norm, h_norm, ocr_confidence, present_flag]
    """
    boxes, (h, w) = detect_fields(img)

    node_feats = []
    for field in FIELD_NAMES:
        if field in boxes:
            x, y, bw, bh, conf = boxes[field]
            node_feats.append([x / w, y / h, bw / w, bh / h, conf, 1.0])
        else:
            node_feats.append([0, 0, 0, 0, 0, 0.0])

    x_tensor = torch.tensor(node_feats, dtype=torch.float)
    edge_index = build_fully_connected_edges(NUM_NODES)
    return x_tensor, edge_index, boxes


# ---------------------------------------------------------------------
# GRAD-CAM VISUALISATION HELPERS
# ---------------------------------------------------------------------
def overlay_heatmap_on_image(original_img, heatmap):
    """Blend the Grad-CAM heatmap on top of the original document image."""
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
    return overlay


def image_to_base64(img_bgr):
    """Encode an OpenCV (BGR) image as a base64 PNG string for the web page."""
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode("utf-8")
