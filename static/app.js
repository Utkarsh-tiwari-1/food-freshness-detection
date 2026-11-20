const fileInput = document.querySelector("#image-input");
const dropArea = document.querySelector("#drop-area");
const emptyUpload = document.querySelector("#upload-empty");
const previewWrap = document.querySelector("#preview-wrap");
const imagePreview = document.querySelector("#image-preview");
const analyzeButton = document.querySelector("#analyze-button");
const form = document.querySelector("#analyzer-form");
const errorBox = document.querySelector("#form-error");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");
const temperature = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperature-value");
const status = document.querySelector("#model-status");

let selectedFile = null;
let previewUrl = null;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function setFile(file) {
  showError("");
  if (!file) return;
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Choose a JPG, PNG or WEBP image.");
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    showError("That image is over 12 MB. Choose a smaller file.");
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  emptyUpload.hidden = true;
  previewWrap.hidden = false;
  analyzeButton.disabled = false;
}

function openPicker() {
  fileInput.click();
}

fileInput.addEventListener("change", () => setFile(fileInput.files?.[0]));
dropArea.addEventListener("click", (event) => {
  if (!event.target.closest("#change-photo")) openPicker();
});
dropArea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openPicker();
  }
});
document.querySelector("#change-photo").addEventListener("click", (event) => {
  event.stopPropagation();
  openPicker();
});
for (const name of ["dragenter", "dragover"]) {
  dropArea.addEventListener(name, (event) => {
    event.preventDefault();
    dropArea.classList.add("dragging");
  });
}
for (const name of ["dragleave", "drop"]) {
  dropArea.addEventListener(name, (event) => {
    event.preventDefault();
    dropArea.classList.remove("dragging");
  });
}
dropArea.addEventListener("drop", (event) => setFile(event.dataTransfer?.files?.[0]));

temperature.addEventListener("input", () => {
  temperatureValue.textContent = temperature.value;
  const percent = ((Number(temperature.value) - Number(temperature.min)) / (Number(temperature.max) - Number(temperature.min))) * 100;
  temperature.style.background = `linear-gradient(to right, #8fae72 ${percent}%, #e7ece3 ${percent}%)`;
});
temperature.dispatchEvent(new Event("input"));

function setModelStatus(ready, message) {
  status.querySelector(".status-dot").className = `status-dot ${ready ? "ready" : "unavailable"}`;
  status.querySelector("span:last-child").textContent = message;
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    setModelStatus(health.model_ready, health.model_ready ? "ResNet18 ready" : "Model setup needed");
  } catch {
    setModelStatus(false, "API unavailable");
  }
}

const titles = { fresh: "Fresh", aging: "Aging", spoiled: "Spoiled" };
const summaries = {
  fresh: "The model sees visual cues most consistent with fresh food.",
  aging: "The model sees signs of aging. Consider prioritizing this item.",
  spoiled: "The model sees visual cues consistent with spoilage. A photo alone cannot confirm safety.",
};

function renderResult(data) {
  resultEmpty.hidden = true;
  resultContent.hidden = false;
  const confidence = Math.round(data.confidence * 100);
  const days = data.shelf_life_days;
  const estimate = data.freshness === "spoiled"
    ? "0 days"
    : `${days.low}–${days.high} <small>days</small>`;
  const rows = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([label, probability]) => `<div class="probability-row"><span>${titles[label] || label}</span><div class="confidence-track"><div class="confidence-fill" style="width:${Math.round(probability * 100)}%"></div></div><b>${Math.round(probability * 100)}%</b></div>`)
    .join("");
  resultContent.innerHTML = `
    <div class="result-topline"><span class="section-kicker">FRESHNESS READING</span><span class="confidence-pill">${confidence}% confidence</span></div>
    <h3 class="freshness-title ${data.freshness}">${titles[data.freshness] || "Unclassified"}</h3>
    <p class="result-summary">${summaries[data.freshness] || "Visual estimate from the trained image model."}</p>
    <div class="confidence-track"><div class="confidence-fill" style="width:${confidence}%"></div></div>
    <div class="confidence-caption"><span>Model confidence</span><span>${confidence}%</span></div>
    <div class="estimate-card"><div class="estimate-label">Illustrative quality window</div><div class="estimate-number">${estimate}</div><div class="estimate-caption">At ${Number(data.temperature_c).toFixed(0)}°C · ${data.food_type.replaceAll("_", " ")}</div></div>
    <p class="result-note ${data.temperature_c > 4 ? "warning" : ""}">${data.estimate_note}</p>
    <div class="probability-list">${rows}</div>
    <p class="result-note">${data.safety_note}</p>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) return;
  showError("");
  analyzeButton.disabled = true;
  analyzeButton.classList.add("loading");
  analyzeButton.querySelector("span:first-child").textContent = "Analyzing";
  const payload = new FormData();
  payload.append("file", selectedFile);
  payload.append("food_type", document.querySelector("#food-type").value);
  payload.append("temperature_c", temperature.value);
  try {
    const response = await fetch("/api/predict", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not analyze this image.");
    renderResult(data);
  } catch (error) {
    showError(error.message || "The service could not be reached. Check that the API is running.");
  } finally {
    analyzeButton.disabled = !selectedFile;
    analyzeButton.classList.remove("loading");
    analyzeButton.querySelector("span:first-child").textContent = "Analyze freshness";
  }
});

loadHealth();
