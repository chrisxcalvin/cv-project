// DeepCheck web UI logic. Vanilla JS, no framework/CDN dependency.

const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 68; // matches r=68 in the SVG

const els = {
  video: document.getElementById("video"),
  canvas: document.getElementById("captureCanvas"),
  captureBtn: document.getElementById("captureBtn"),
  resetBtn: document.getElementById("resetBtn"),
  cameraCaption: document.getElementById("cameraCaption"),
  verdictGlow: document.getElementById("verdictGlow"),
  scanOverlay: document.getElementById("scanOverlay"),
  historyStrip: document.getElementById("historyStrip"),

  gaugeArc: document.getElementById("gaugeArc"),
  gaugePct: document.getElementById("gaugePct"),
  gaugeVerdict: document.getElementById("gaugeVerdict"),
  spoofBadge: document.getElementById("spoofBadge"),
  methodLine: document.getElementById("methodLine"),
  featureBars: document.getElementById("featureBars"),

  imgCrop: document.getElementById("imgCrop"),
  imgEdges: document.getElementById("imgEdges"),
  imgOrb: document.getElementById("imgOrb"),
  imgFft: document.getElementById("imgFft"),

  demoModeToggle: document.getElementById("demoModeToggle"),
  glassesModeToggle: document.getElementById("glassesModeToggle"),

  navTabs: document.querySelectorAll(".nav-tab"),
  viewLive: document.getElementById("view-live"),
  viewAnalytics: document.getElementById("view-analytics"),

  statTotal: document.getElementById("statTotal"),
  statAccept: document.getElementById("statAccept"),
  statReject: document.getElementById("statReject"),
  statChallenge: document.getElementById("statChallenge"),
  attemptsBody: document.getElementById("attemptsBody"),

  modal: document.getElementById("challengeModal"),
  modalVideo: document.getElementById("modalVideo"),
  challengeIcon: document.getElementById("challengeIcon"),
  challengeTitle: document.getElementById("challengeTitle"),
  challengeInstruction: document.getElementById("challengeInstruction"),
  challengeCaptureBtn: document.getElementById("challengeCaptureBtn"),
  challengeCancelBtn: document.getElementById("challengeCancelBtn"),
  challengeNote: document.getElementById("challengeNote"),
};

let mediaStream = null;
let confidenceHistory = [];

const ICONS = {
  eye: "\u{1F441}", "arrow-left": "←", "arrow-right": "→", mouth: "\u{1F444}",
};

// ---------------------------------------------------------------- camera --

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    els.video.srcObject = mediaStream;
    els.modalVideo.srcObject = mediaStream;
    els.cameraCaption.textContent = "Position your face in frame";
  } catch (err) {
    els.cameraCaption.textContent = "Camera access denied or unavailable";
    els.captureBtn.disabled = true;
  }
}

function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

async function captureBurst(videoEl, count = 12, intervalMs = 80) {
  // Capture a short burst of frames instead of one photo, so the user
  // just has to perform the gesture (blink / turn / open mouth) SOMEWHERE
  // in a ~1s window rather than hit one exact millisecond - see the
  // matching comment in webapp.py's /api/challenge.
  const frames = [];
  for (let i = 0; i < count; i++) {
    frames.push(captureFrame(videoEl));
    await sleep(intervalMs);
  }
  return frames;
}

function captureFrame(videoEl) {
  const canvas = els.canvas;
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  // Un-mirror before sending: the <video> is CSS-flipped for a natural
  // "mirror" preview, but the backend should see the frame the webcam
  // actually captured, not a flipped copy.
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

// ---------------------------------------------------------------- verdict UI --

function setVerdictGlow(state) {
  els.verdictGlow.className = "verdict-glow " + state;
}

function setGauge(pct, label) {
  const clamped = Math.max(0, Math.min(1, pct));
  const offset = GAUGE_CIRCUMFERENCE * (1 - clamped);
  els.gaugeArc.style.strokeDashoffset = offset;
  els.gaugePct.textContent = Math.round(clamped * 100) + "%";
  els.gaugeVerdict.textContent = label;

  let color = "var(--accent)";
  if (label === "ACCEPT") color = "var(--ok)";
  else if (label === "REJECT") color = "var(--bad)";
  else if (label === "CHALLENGE REQUIRED") color = "var(--warn)";
  els.gaugeArc.style.stroke = color;
}

function renderFeatureBars(passive) {
  const rows = els.featureBars.querySelectorAll(".feature-row");
  const values = {
    texture: passive.texture_var, orb: passive.orb_keypoints,
    fft: passive.fft_ratio, edge: passive.edge_density,
  };
  rows.forEach((row) => {
    const key = row.dataset.feature;
    const sub = passive.sub_scores[key];
    row.querySelector(".feature-fill").style.width = Math.round(sub * 100) + "%";
    row.querySelector(".feature-val").textContent =
      typeof values[key] === "number" ? values[key].toFixed(key === "fft" || key === "edge" ? 4 : 1) : "--";
  });
  els.methodLine.textContent = "Method: " + passive.method + (passive.spoof_type ? "  |  Spoof type: " + passive.spoof_type.replace(/_/g, " ") : "");
}

function pushHistory(verdict) {
  confidenceHistory.push(verdict);
  if (confidenceHistory.length > 24) confidenceHistory.shift();
  els.historyStrip.innerHTML = "";
  confidenceHistory.forEach((v) => {
    const bar = document.createElement("div");
    const cls = v === "ACCEPT" ? "accept" : v === "REJECT" ? "reject" : "pending";
    bar.className = "bar " + cls;
    bar.style.height = (30 + Math.random() * 4) + "px"; // subtle organic variation
    els.historyStrip.appendChild(bar);
  });
}

function resetResultPanel() {
  setGauge(0, "AWAITING SCAN");
  setVerdictGlow("");
  els.spoofBadge.hidden = true;
  els.methodLine.textContent = "Method: --";
  els.featureBars.querySelectorAll(".feature-row").forEach((row) => {
    row.querySelector(".feature-fill").style.width = "0%";
    row.querySelector(".feature-val").textContent = "--";
  });
  [els.imgCrop, els.imgEdges, els.imgOrb, els.imgFft].forEach((img) => (img.src = ""));
  els.cameraCaption.textContent = "Position your face in frame";
}

// ---------------------------------------------------------------- verify flow --

async function runVerify() {
  els.captureBtn.disabled = true;
  els.cameraCaption.textContent = "Analyzing...";
  setVerdictGlow("scanning");

  try {
    const dataUrl = captureFrame(els.video);
    const res = await fetch("/api/verify", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl, demo_mode: els.demoModeToggle.checked }),
    });
    const data = await res.json();

    if (data.error) {
      els.cameraCaption.textContent = data.message || "Verification failed.";
      setVerdictGlow("");
      els.captureBtn.disabled = false;
      return;
    }

    applyPassiveResult(data);

    if (data.verdict === "PENDING") {
      openChallengeModal(data.challenge);
    } else {
      finalizeVerdict(data.verdict, data.confidence);
    }
  } catch (err) {
    els.cameraCaption.textContent = "Network error - is the server running?";
    setVerdictGlow("");
  }
  els.captureBtn.disabled = false;
}

function applyPassiveResult(data) {
  const passive = data.passive;
  setGauge(passive.real_confidence, data.verdict === "PENDING" ? "CHALLENGE REQUIRED" : data.verdict);
  renderFeatureBars(passive);
  if (passive.spoof_type) {
    els.spoofBadge.hidden = false;
    els.spoofBadge.textContent = "Likely: " + passive.spoof_type.replace(/_/g, " ");
  } else {
    els.spoofBadge.hidden = true;
  }
  els.imgCrop.src = data.face_crop || "";
  els.imgEdges.src = passive.images.edges || "";
  els.imgOrb.src = passive.images.orb_keypoints_img || "";
  els.imgFft.src = passive.images.fft_spectrum || "";
}

function finalizeVerdict(verdict, confidence) {
  setVerdictGlow(verdict === "ACCEPT" ? "accept" : "reject");
  setGauge(confidence, verdict);
  els.cameraCaption.textContent = verdict === "ACCEPT" ? "Identity verified" : "Verification rejected";
  pushHistory(verdict);
}

// ---------------------------------------------------------------- challenge modal --

function openChallengeModal(challenge) {
  els.modal.hidden = false;
  els.challengeIcon.textContent = ICONS[challenge.icon] || "✨";
  els.challengeTitle.textContent = challenge.label;
  els.challengeInstruction.textContent = challenge.instruction;
  els.challengeNote.textContent = "";
  setVerdictGlow("pending");
}

function closeChallengeModal() {
  els.modal.hidden = true;
}

async function runChallengeCapture() {
  els.challengeCaptureBtn.disabled = true;
  els.challengeNote.style.color = "var(--text-dim)";
  els.challengeNote.textContent = "Hold the gesture...";

  try {
    const frames = await captureBurst(els.modalVideo);
    els.challengeNote.textContent = "Analyzing...";
    const res = await fetch("/api/challenge", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images: frames, glasses_mode: els.glassesModeToggle.checked }),
    });
    const data = await res.json();

    if (data.error) {
      els.challengeNote.style.color = "var(--bad)";
      els.challengeNote.textContent = data.message || "Challenge failed.";
      els.challengeCaptureBtn.disabled = false;
      return;
    }

    if (data.retry) {
      els.challengeNote.style.color = "var(--warn)";
      els.challengeNote.textContent = data.message;
      els.challengeCaptureBtn.disabled = false;
      return;
    }

    closeChallengeModal();
    finalizeVerdict(data.verdict, data.confidence);
  } catch (err) {
    els.challengeNote.style.color = "var(--bad)";
    els.challengeNote.textContent = "Network error - is the server running?";
  }
  els.challengeCaptureBtn.disabled = false;
}

// ---------------------------------------------------------------- analytics --

async function loadAnalytics() {
  const res = await fetch("/api/analytics");
  const data = await res.json();
  const stats = data.stats;

  els.statTotal.textContent = stats.total;
  els.statAccept.textContent = Math.round(stats.accept_rate * 100) + "%";
  els.statReject.textContent = Math.round(stats.reject_rate * 100) + "%";
  els.statChallenge.textContent = Math.round(stats.challenge_rate * 100) + "%";

  els.attemptsBody.innerHTML = "";
  data.recent.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.timestamp || ""}</td>
      <td><span class="tag ${r.verdict}">${r.verdict}</span></td>
      <td>${r.real_confidence != null ? (r.real_confidence * 100).toFixed(1) + "%" : "--"}</td>
      <td>${r.method || "--"}</td>
      <td>${r.spoof_type ? r.spoof_type.replace(/_/g, " ") : "--"}</td>
      <td>${r.challenge_type || "--"}</td>
      <td>${r.reason || "--"}</td>
    `;
    els.attemptsBody.appendChild(tr);
  });
}

// ---------------------------------------------------------------- nav --

els.navTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    els.navTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const view = tab.dataset.view;
    els.viewLive.hidden = view !== "live";
    els.viewAnalytics.hidden = view !== "analytics";
    if (view === "analytics") loadAnalytics();
  });
});

// ---------------------------------------------------------------- wire up --

els.captureBtn.addEventListener("click", runVerify);
els.resetBtn.addEventListener("click", resetResultPanel);
els.challengeCaptureBtn.addEventListener("click", runChallengeCapture);
els.challengeCancelBtn.addEventListener("click", () => {
  closeChallengeModal();
  els.cameraCaption.textContent = "Challenge cancelled - capture again to retry";
  setVerdictGlow("");
});

startCamera();
