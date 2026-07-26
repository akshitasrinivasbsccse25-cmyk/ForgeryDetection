// script.js
// Handles: image preview, calling /upload then /predict, and rendering
// the real prediction + Grad-CAM result returned by the Flask backend.

const fileInput = document.getElementById("fileInput");
const previewImg = document.getElementById("previewImg");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const detectBtn = document.getElementById("detectBtn");
const statusMsg = document.getElementById("statusMsg");

const resultCard = document.getElementById("resultCard");
const predictionValue = document.getElementById("predictionValue");
const confidenceValue = document.getElementById("confidenceValue");
const timeValue = document.getElementById("timeValue");

const heatmapCard = document.getElementById("heatmapCard");
const heatmapImg = document.getElementById("heatmapImg");

let uploadedFilePath = null;

// --- Show a preview of the selected image ---
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.style.display = "block";
    previewPlaceholder.style.display = "none";
  };
  reader.readAsDataURL(file);

  uploadFile(file);
});

// --- Upload the file to the Flask backend ---
async function uploadFile(file) {
  statusMsg.textContent = "Uploading image...";
  detectBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData
    });
    const data = await response.json();

    if (data.error) {
      statusMsg.textContent = "Error: " + data.error;
      return;
    }

    uploadedFilePath = data.filepath;
    statusMsg.textContent = "Image uploaded. Click 'Detect Forgery' to run the model.";
    detectBtn.disabled = false;
  } catch (err) {
    statusMsg.textContent = "Upload failed: " + err;
  }
}

// --- Run real model inference via /predict ---
detectBtn.addEventListener("click", async () => {
  if (!uploadedFilePath) return;

  statusMsg.textContent = "Running hybrid GAT + DCT-CNN inference...";
  detectBtn.disabled = true;
  resultCard.style.display = "none";
  heatmapCard.style.display = "none";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filepath: uploadedFilePath })
    });
    const data = await response.json();

    if (data.error) {
      statusMsg.textContent = "Error: " + data.error;
      detectBtn.disabled = false;
      return;
    }

    // Show prediction results
    predictionValue.textContent = data.prediction;
    predictionValue.style.color = (data.prediction === "Forged") ? "#c62828" : "#0d47a1";
    confidenceValue.textContent = data.confidence;
    timeValue.textContent = data.processing_time;
    resultCard.style.display = "block";

    // Show Grad-CAM heatmap if available
    if (data.heatmap) {
      heatmapImg.src = "data:image/png;base64," + data.heatmap;
      heatmapCard.style.display = "block";
    }

    statusMsg.textContent = "Detection complete.";
  } catch (err) {
    statusMsg.textContent = "Prediction failed: " + err;
  } finally {
    detectBtn.disabled = false;
  }
});
