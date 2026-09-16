const LANG_NAMES = {
  hi: "Hindi",
  kn: "Kannada",
  mr: "Marathi",
  ta: "Tamil",
  te: "Telugu",
  as: "Assamese",
  bn: "Bengali",
  brx: "Bodo",
  doi: "Dogri",
  gu: "Gujarati",
  kok: "Konkani",
  ks: "Kashmiri",
  mai: "Maithili",
  ml: "Malayalam",
  mni: "Manipuri",
  ne: "Nepali",
  or: "Odia",
  pa: "Punjabi",
  sa: "Sanskrit",
  sat: "Santali",
  sd: "Sindhi",
  ur: "Urdu",
};

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const hint = document.getElementById("hint");
const result = document.getElementById("result");

let mediaStream = null;
let audioContext = null;
let processor = null;
let source = null;
let recording = false;
let samples = [];

async function initMic() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    hint.textContent = "Ready — press Start and speak.";
    startBtn.disabled = false;
  } catch (err) {
    startBtn.disabled = true;
    stopBtn.disabled = true;
    showError("Microphone access denied.");
  }
}

function showError(msg) {
  result.innerHTML = `<div class="status error">${msg}</div>`;
}

function showLoading() {
  result.innerHTML = `<div class="status pulse">Identifying…</div>`;
}

function encodeWav(float32, sampleRate) {
  const buffer = new ArrayBuffer(44 + float32.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + float32.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, float32.length * 2, true);
  let off = 44;
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const len = Math.round(buffer.length / ratio);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    out[i] = buffer[Math.round(i * ratio)];
  }
  return out;
}

async function startRecording() {
  if (!mediaStream || recording) return;
  recording = true;
  samples = [];
  startBtn.disabled = true;
  stopBtn.disabled = false;
  hint.textContent = "Recording… press Stop when finished.";

  audioContext = new AudioContext();
  source = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (e) => {
    if (recording) samples.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(audioContext.destination);
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  hint.textContent = "Ready — press Start and speak.";

  processor.disconnect();
  source.disconnect();
  const rate = audioContext.sampleRate;
  await audioContext.close();

  const total = samples.reduce((n, c) => n + c.length, 0);
  if (total < rate * 0.3) {
    showError("Too short — speak for at least half a second.");
    return;
  }

  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of samples) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  const pcm = downsample(merged, rate, 16000);
  const wav = encodeWav(pcm, 16000);
  await identify(wav);
}

async function identify(blob) {
  showLoading();
  const form = new FormData();
  form.append("audio", blob, "speech.wav");
  try {
    const res = await fetch("/identify", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      const detail = data.detail;
      throw new Error(typeof detail === "string" ? detail : res.statusText);
    }
    renderResult(data);
  } catch (err) {
    showError(err.message || "Identification failed.");
  }
}

function renderResult(data) {
  const code = data.language;
  if (!code) {
    showError("Could not identify language. Try speaking more clearly.");
    return;
  }

  const name = LANG_NAMES[code] || code.toUpperCase();
  const margin = data.margin != null ? data.margin.toFixed(3) : "—";
  const candidates = data.top_candidates || [];
  const scores = candidates.map((c) => c.score);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const range = max - min || 1;

  const rows = candidates
    .map((c, i) => {
      const pct = ((c.score - min) / range) * 100;
      return `
        <div class="score-row ${i === 0 ? "top" : ""}">
          <span>${c.language}</span>
          <div class="bar-wrap"><div class="bar" style="width:${pct}%"></div></div>
          <span class="val">${c.score.toFixed(2)}</span>
        </div>`;
    })
    .join("");

  result.innerHTML = `
    <div class="result-label">Detected language</div>
    <div class="language">${name}</div>
    <div class="language-code">${code} · margin ${margin}</div>
    <div class="scores">${rows}</div>`;
}

startBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
startBtn.disabled = true;
stopBtn.disabled = true;
initMic();
