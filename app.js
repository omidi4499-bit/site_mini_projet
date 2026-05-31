const referenceCatalog = {
  card_w: { label: 'Carte bancaire (largeur)', lengthCm: 5.398 },
  card_h: { label: 'Carte bancaire (longueur)', lengthCm: 8.56 },
  a4_w: { label: 'Feuille A4 (largeur)', lengthCm: 21.0 },
  a4_h: { label: 'Feuille A4 (longueur)', lengthCm: 29.7 },
  a5_w: { label: 'Feuille A5 (largeur)', lengthCm: 14.8 },
  a5_h: { label: 'Feuille A5 (longueur)', lengthCm: 21.0 },
  coin_2dh: { label: 'Piece 2 DH (diametre)', lengthCm: 2.6 },
  coin_10dh: { label: 'Piece 10 DH (diametre)', lengthCm: 2.69 }
};

const startBtn = document.getElementById('startBtn');
const welcomeView = document.getElementById('welcomeView');
const measureView = document.getElementById('measureView');

const imageInput = document.getElementById('imageInput');
const openCameraBtn = document.getElementById('openCameraBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const cameraModal = document.getElementById('cameraModal');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');

const referenceSelect = document.getElementById('referenceSelect');
const referenceInfo = document.getElementById('referenceInfo');
const unitSelect = document.getElementById('unitSelect');
const computeBtn = document.getElementById('computeBtn');
const clearAllBtn = document.getElementById('clearAll');
const stepHelp = document.getElementById('stepHelp');
const statusNode = document.getElementById('status');
const noteText = document.getElementById('noteText');

const placeholder = document.getElementById('placeholder');
const canvas = document.getElementById('measureCanvas');
const ctx = canvas.getContext('2d');

const widthValue = document.getElementById('widthValue');
const lengthValue = document.getElementById('lengthValue');
const areaValue = document.getElementById('areaValue');
const precisionValue = document.getElementById('precisionValue');

const modeLimits = {
  reference: 2,
  width: 2,
  length: 2
};

const modeColors = {
  reference: '#e27a3f',
  width: '#117257',
  length: '#1e5cb8'
};

const points = {
  reference: [],
  width: [],
  length: []
};

let image = null;
let imageDataUrl = '';
let cameraStream = null;

function formatNumber(value, digits = 2) {
  return Number(value).toLocaleString('fr-FR', {
    maximumFractionDigits: digits
  });
}

function setStatus(message) {
  statusNode.textContent = message;
}

function unitFactor(unit) {
  if (unit === 'mm') {
    return 10;
  }
  if (unit === 'm') {
    return 0.01;
  }
  if (unit === 'in') {
    return 1 / 2.54;
  }
  return 1;
}

function getSelectedReference() {
  return referenceCatalog[referenceSelect.value] || referenceCatalog.card_w;
}

function updateReferenceInfo() {
  const ref = getSelectedReference();
  referenceInfo.value = `${formatNumber(ref.lengthCm, 3)} cm`;
}

function clearResults() {
  widthValue.textContent = '-';
  lengthValue.textContent = '-';
  areaValue.textContent = '-';
  precisionValue.textContent = '-';
  noteText.textContent = "La reference doit etre sur le meme plan que l'objet.";
}

function currentMode() {
  if (points.reference.length < modeLimits.reference) {
    return 'reference';
  }
  if (points.width.length < modeLimits.width) {
    return 'width';
  }
  if (points.length.length < modeLimits.length) {
    return 'length';
  }
  return 'done';
}

function updateHelp() {
  if (!image) {
    stepHelp.textContent = 'Charge une image pour commencer.';
    return;
  }

  const mode = currentMode();
  if (mode === 'reference') {
    stepHelp.textContent = `Reference: ${points.reference.length}/2 points places.`;
    return;
  }

  if (mode === 'width') {
    stepHelp.textContent = `Largeur: ${points.width.length}/2 points places.`;
    return;
  }

  if (mode === 'length') {
    stepHelp.textContent = `Longueur: ${points.length.length}/2 points places.`;
    return;
  }

  stepHelp.textContent = 'Points complets. Lance la mesure ou recommence si besoin.';
}

function resetMeasurements() {
  points.reference = [];
  points.width = [];
  points.length = [];
  clearResults();
  draw();
  updateHelp();
}

function loadImageFromDataUrl(dataUrl) {
  if (!dataUrl) {
    return;
  }

  imageDataUrl = dataUrl;
  const img = new Image();
  img.onload = () => {
    image = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    placeholder.style.display = 'none';
    resetMeasurements();
    setStatus('Image chargee. Place d\'abord les 2 points de reference.');
  };
  img.src = imageDataUrl;
}

function loadImage(file) {
  if (!file) {
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    loadImageFromDataUrl(event.target.result);
  };
  reader.readAsDataURL(file);
}

function getImageCoords(mouseEvent) {
  const rect = canvas.getBoundingClientRect();
  const x = (mouseEvent.clientX - rect.left) * (canvas.width / rect.width);
  const y = (mouseEvent.clientY - rect.top) * (canvas.height / rect.height);
  return { x, y };
}

function drawPoint(point, color, label) {
  ctx.beginPath();
  ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  ctx.beginPath();
  ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  if (!label) {
    return;
  }

  ctx.font = '600 16px "IBM Plex Mono"';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(255,255,255,0.95)';
  ctx.strokeText(label, point.x + 8, point.y + 8);
  ctx.fillStyle = color;
  ctx.fillText(label, point.x + 8, point.y + 8);
}

function drawSegment(segmentPoints, color, label) {
  if (segmentPoints.length < 2) {
    return;
  }

  ctx.beginPath();
  ctx.moveTo(segmentPoints[0].x, segmentPoints[0].y);
  ctx.lineTo(segmentPoints[1].x, segmentPoints[1].y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.stroke();

  const midX = (segmentPoints[0].x + segmentPoints[1].x) / 2;
  const midY = (segmentPoints[0].y + segmentPoints[1].y) / 2;

  ctx.font = '600 15px "IBM Plex Mono"';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.lineWidth = 4;
  ctx.strokeStyle = 'rgba(255,255,255,0.95)';
  ctx.strokeText(label, midX, midY - 8);
  ctx.fillStyle = color;
  ctx.fillText(label, midX, midY - 8);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!image) {
    return;
  }

  ctx.drawImage(image, 0, 0);

  points.reference.forEach((point, index) => drawPoint(point, modeColors.reference, `R${index + 1}`));
  drawSegment(points.reference, modeColors.reference, 'Ref');

  points.width.forEach((point, index) => drawPoint(point, modeColors.width, `L${index + 1}`));
  drawSegment(points.width, modeColors.width, 'Largeur');

  points.length.forEach((point, index) => drawPoint(point, modeColors.length, `G${index + 1}`));
  drawSegment(points.length, modeColors.length, 'Longueur');
}

function pushPoint(point) {
  const mode = currentMode();
  if (mode === 'done') {
    resetMeasurements();
  }

  const nextMode = currentMode();
  points[nextMode].push(point);

  draw();
  updateHelp();
}

function distance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function validateBeforeCompute() {
  if (!image || !imageDataUrl) {
    throw new Error('Charge une image avant de calculer.');
  }

  if (points.reference.length !== 2) {
    throw new Error('Il faut 2 points pour la reference.');
  }

  if (points.width.length !== 2) {
    throw new Error('Il faut 2 points pour la largeur.');
  }

  if (points.length.length !== 2) {
    throw new Error('Il faut 2 points pour la longueur.');
  }
}

function precisionLabel(refPixels) {
  if (refPixels >= 160) {
    return 'Elevee';
  }
  if (refPixels >= 90) {
    return 'Moyenne';
  }
  return 'Faible';
}

// --- Nouvelle fonction : appel au backend de mesure 3D ---
async function callBackendMeasure() {
  if (!imageDataUrl) return false;

  const payload = {
    image_data_url: imageDataUrl,
    image_width: canvas.width,
    image_height: canvas.height,
    unit: unitSelect.value || 'cm',
    reference_points: points.reference.map(p => ({ x: p.x, y: p.y })),
    reference_length_cm: getSelectedReference().lengthCm,
    width_points: points.width.map(p => ({ x: p.x, y: p.y })),
    height_points: points.length.map(p => ({ x: p.x, y: p.y }))
  };

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    const response = await fetch('/api/measure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    clearTimeout(timeoutId);
    if (!response.ok) return false;

    const result = await response.json();
    if (result.error) return false;

    const unit = result.unit || 'cm';

    widthValue.textContent = `${formatNumber(result.width, 2)} ${unit}`;
    lengthValue.textContent = `${formatNumber(result.height, 2)} ${unit}`;
    areaValue.textContent = `${formatNumber(result.area, 2)} ${unit}^2`;
    precisionValue.textContent = result.precision_label || '-';
    noteText.textContent = result.note || '';

    setStatus(`Mesure terminee (analyse 3D): ${formatNumber(result.width, 1)} x ${formatNumber(result.height, 1)} ${unit}.`);
    return true;
  } catch (err) {
    return false;
  }
}

// --- Fonction de calcul modifiée : backend first, fallback local ---
async function computeDimensions() {
  // Essayer d’abord la mesure via le backend
  const backendOk = await callBackendMeasure();
  if (backendOk) return;

  // Fallback : calcul côté client (comportement d’origine)
  try {
    validateBeforeCompute();
  } catch (error) {
    setStatus(error.message);
    return;
  }

  const ref = getSelectedReference();
  const refPx = distance(points.reference[0], points.reference[1]);
  if (refPx < 8) {
    setStatus('Reference trop courte. Zoome ou rapproche la camera.');
    return;
  }

  const scaleCm = ref.lengthCm / refPx;
  const widthPx = distance(points.width[0], points.width[1]);
  const lengthPx = distance(points.length[0], points.length[1]);

  const widthCm = widthPx * scaleCm;
  const lengthCm = lengthPx * scaleCm;
  const areaCm2 = widthCm * lengthCm;

  const unit = unitSelect.value || 'cm';
  const factor = unitFactor(unit);

  widthValue.textContent = `${formatNumber(widthCm * factor, 2)} ${unit}`;
  lengthValue.textContent = `${formatNumber(lengthCm * factor, 2)} ${unit}`;
  areaValue.textContent = `${formatNumber(areaCm2 * factor * factor, 2)} ${unit}^2`;
  precisionValue.textContent = precisionLabel(refPx);

  noteText.textContent = "La reference et l'objet doivent etre sur le meme plan.";
  setStatus(`Mesure terminee: ${formatNumber(widthCm * factor, 1)} x ${formatNumber(lengthCm * factor, 1)} ${unit}.`);
}

// --- Écouteurs d’événements (inchangés) ---
startBtn.addEventListener('click', () => {
  welcomeView.classList.add('hidden');
  measureView.classList.remove('hidden');
  setStatus('En attente d\'image...');
  updateHelp();
});

referenceSelect.addEventListener('change', () => {
  updateReferenceInfo();
  clearResults();
  if (image) {
    setStatus('Reference mise a jour. Recalcule pour actualiser les mesures.');
  }
});

imageInput.addEventListener('change', (event) => {
  loadImage(event.target.files?.[0]);
});

openCameraBtn.addEventListener('click', () => {
  openCamera();
});

closeCameraBtn.addEventListener('click', () => {
  closeCamera();
});

captureBtn.addEventListener('click', () => {
  capturePhoto();
});

cameraModal.addEventListener('click', (event) => {
  if (event.target === cameraModal) {
    closeCamera();
  }
});

canvas.addEventListener('click', (event) => {
  if (!image) {
    return;
  }

  pushPoint(getImageCoords(event));
});

computeBtn.addEventListener('click', () => {
  computeDimensions();
});

clearAllBtn.addEventListener('click', () => {
  if (!image) {
    return;
  }

  resetMeasurements();
  setStatus('Mesures reinitialisees.');
});

updateReferenceInfo();
setStatus('Clique sur "Demarrer la mesure" pour commencer.');
updateHelp();