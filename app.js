// ChantScanner: photograph an Anglican chant, get editable ABC, see and hear it.

const MAX_SIDE = 1024; // longest edge before the model's own sizing
// LFM2-VL's image grid: sides multiples of 32 px (one image token each 32x32), 64 to 256 tokens.
const ALIGN = 32;
const MIN_PIXELS = 64 * ALIGN * ALIGN;
const MAX_PIXELS = 256 * ALIGN * ALIGN;

const $ = (id) => document.getElementById(id);
const worker = new Worker("./worker.js", { type: "module" });

let modelState = "idle"; // idle | loading | ready | busy
let pendingImage = null; // { data, width, height } scaled for the model
let transpose = 0;
let voices = "SATB";
let synthControl = null;
let output = "";
let tokens = 0;

// ---------------------------------------------------------------- model

function setStatus(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function updateUI() {
  const busy = modelState === "busy";
  $("load-button").hidden = modelState === "ready" || busy;
  $("load-button").disabled = modelState !== "idle";
  $("transcribe-button").disabled = modelState !== "ready" || !pendingImage;
  $("stop-button").hidden = !busy;
  $("reading").hidden = !busy;
  $("scanline").hidden = !busy;
}

function loadModel() {
  if (modelState !== "idle") return;
  modelState = "loading";
  updateUI();
  setStatus("Starting…");
  $("progress").hidden = false;
  $("progress").classList.add("indeterminate");
  worker.postMessage({ type: "load" });
}

worker.addEventListener("message", ({ data: msg }) => {
  if (msg.type === "progress") {
    $("progress").classList.remove("indeterminate");
    $("progress-fill").style.width = `${(100 * msg.loaded) / msg.total}%`;
    const mb = (x) => (x / 1e6).toFixed(0);
    setStatus(`Downloading ${mb(msg.loaded)} / ${mb(msg.total)} MB`);
    if (msg.loaded >= msg.total) {
      $("progress").classList.add("indeterminate");
      setStatus("Preparing the model…");
    }
  } else if (msg.type === "ready") {
    modelState = "ready";
    $("progress").hidden = true;
    const gpu = msg.device === "webgpu";
    const cpu = `CPU · ${msg.threads} thread${msg.threads === 1 ? "" : "s"}`;
    $("device-pill").textContent = `● ready · ${gpu ? "WebGPU" : cpu}`;
    $("device-pill").hidden = false;
    const why = {
      "not-secure": " Only one thread because this page isn't on https (or localhost), so the browser blocks multi-threading.",
      "not-isolated": " Only one thread: reload the page once so its service worker can switch on multi-threading.",
    };
    setStatus(
      (gpu ? "Running on your GPU." : `No WebGPU here, so it runs on the CPU (${cpu}).`) +
        (!gpu && msg.whySingle ? why[msg.whySingle] : "") +
        (pendingImage ? "" : " Add a picture of a chant."),
    );
    updateUI();
    if (pendingImage) transcribe();
  } else if (msg.type === "token") {
    output += msg.text;
    tokens++;
    $("reading-text").textContent = `Reading the music… ${tokens} tokens`;
    $("abc").value = cleanOutput(output);
    renderThrottled();
  } else if (msg.type === "done") {
    modelState = "ready";
    output = msg.text;
    $("abc").value = cleanOutput(output);
    setStatus(
      `Read in ${msg.seconds.toFixed(1)} s: image ${msg.firstSeconds.toFixed(1)} s, ` +
        `then ${msg.tokensPerSecond.toFixed(1)} tokens/s.`,
    );
    updateUI();
    render(true);
  } else if (msg.type === "error") {
    modelState = modelState === "busy" ? "ready" : "idle";
    $("progress").hidden = true;
    setStatus(`Something went wrong: ${msg.message}`, true);
    updateUI();
  }
});

function transcribe() {
  if (modelState !== "ready" || !pendingImage) return;
  modelState = "busy";
  output = "";
  tokens = 0;
  $("abc").value = "";
  $("reading-text").textContent = "Reading the music…";
  updateUI();
  render(false);
  const { data, width, height } = pendingImage;
  const copy = data.slice(0); // keep pendingImage for a re-run
  worker.postMessage({ type: "transcribe", data: copy, width, height }, [copy]);
}

// Anything before the header (rare) is dropped.
function cleanOutput(text) {
  const i = text.indexOf("X:");
  return (i >= 0 ? text.slice(i) : text).trim() + "\n";
}

// ---------------------------------------------------------------- images

// llama.cpp's calc_size_preserved_ratio / HF smart_resize: what the model was trained on.
function alignedSize(w, h) {
  let wBar = Math.max(ALIGN, Math.round(w / ALIGN) * ALIGN);
  let hBar = Math.max(ALIGN, Math.round(h / ALIGN) * ALIGN);
  if (wBar * hBar > MAX_PIXELS) {
    const beta = Math.sqrt((w * h) / MAX_PIXELS);
    wBar = Math.max(ALIGN, Math.floor(w / beta / ALIGN) * ALIGN);
    hBar = Math.max(ALIGN, Math.floor(h / beta / ALIGN) * ALIGN);
  } else if (wBar * hBar < MIN_PIXELS) {
    const beta = Math.sqrt(MIN_PIXELS / (w * h));
    wBar = Math.ceil((w * beta) / ALIGN) * ALIGN;
    hBar = Math.ceil((h * beta) / ALIGN) * ALIGN;
  }
  return [wBar, hBar];
}

// Shrink in halving steps so thin staff lines are averaged, not skipped (canvas resizing
// alone does not antialias big reductions in every browser).
function resizeTo(source, w, h) {
  let canvas = document.createElement("canvas");
  canvas.width = source.width;
  canvas.height = source.height;
  canvas.getContext("2d").drawImage(source, 0, 0);
  while (canvas.width >= 2 * w && canvas.height >= 2 * h) {
    const half = document.createElement("canvas");
    half.width = Math.round(canvas.width / 2);
    half.height = Math.round(canvas.height / 2);
    const ctx = half.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(canvas, 0, 0, half.width, half.height);
    canvas = half;
  }
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "white"; // transparent screenshots: the model has only seen paper
  ctx.fillRect(0, 0, w, h);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(canvas, 0, 0, w, h);
  return out;
}

// `source`: anything drawImage takes that has width and height (ImageBitmap, canvas).
function useSource(source) {
  const scale = Math.min(1, MAX_SIDE / Math.max(source.width, source.height));
  const [w, h] = alignedSize(Math.round(source.width * scale), Math.round(source.height * scale));
  const canvas = resizeTo(source, w, h);
  pendingImage = { data: canvas.getContext("2d").getImageData(0, 0, w, h).data.buffer, width: w, height: h };
  $("preview").src = canvas.toDataURL("image/png");
  $("viewer").hidden = false;
  $("preview-row").hidden = false;
  $("preview-note").textContent = `${w} × ${h} px → model`;
  updateUI();
  if (modelState === "ready") transcribe();
  else if (modelState === "idle") setStatus("Picture ready. Load the model to read it.");
}

async function useImage(blob) {
  if (!blob || !blob.type.startsWith("image/")) {
    setStatus("That doesn't look like an image.", true);
    return;
  }
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob); // applies the photo's EXIF orientation
  } catch {
    setStatus("Couldn't read that image.", true);
    return;
  }
  useSource(bitmap);
  bitmap.close();
}

async function pasteFromClipboard() {
  try {
    for (const item of await navigator.clipboard.read()) {
      const type = item.types.find((t) => t.startsWith("image/"));
      if (type) return useImage(await item.getType(type));
    }
    setStatus("No image on the clipboard. Copy a picture first, or press Ctrl+V / ⌘V on this page.");
  } catch {
    setStatus("This browser blocked clipboard access. Press Ctrl+V / ⌘V on this page instead.");
  }
}

// ---------------------------------------------------------------- camera

let camStream = null;
let camDevices = [];
let camIndex = 0;

function stopCamera() {
  camStream?.getTracks().forEach((t) => t.stop());
  camStream = null;
}

async function startCamera(deviceId) {
  stopCamera();
  const size = { width: { ideal: 2560 }, height: { ideal: 1440 } };
  const video = deviceId ? { deviceId: { exact: deviceId }, ...size } : { facingMode: { ideal: "environment" }, ...size };
  camStream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  const el = $("cam-video");
  el.srcObject = camStream;
  await new Promise((resolve) => {
    if (el.readyState >= 1) resolve();
    else el.onloadedmetadata = resolve;
  });
  await el.play().catch(() => {});
  layoutStage();
}

// Size the stage to the video's aspect ratio, so frame percentages are video percentages.
function layoutStage() {
  const el = $("cam-video");
  if (!el.videoWidth) return;
  const aspect = el.videoWidth / el.videoHeight;
  const maxW = Math.min(window.innerWidth - 32, 900);
  const maxH = window.innerHeight - 250;
  const w = Math.min(maxW, maxH * aspect);
  $("cam-stage").style.width = `${w}px`;
  $("cam-stage").style.height = `${w / aspect}px`;
  layoutGuide();
}

function guideRect() {
  const stage = $("cam-stage");
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  const ratio = Number($("guide-shape").value); // frame height / width
  let gw = 0.92 * W;
  let gh = gw * ratio;
  if (gh > 0.92 * H) {
    gh = 0.92 * H;
    gw = gh / ratio;
  }
  return { left: (W - gw) / 2, top: (H - gh) / 2, width: gw, height: gh, W };
}

function layoutGuide() {
  const r = guideRect();
  Object.assign($("guide").style, {
    left: `${r.left}px`,
    top: `${r.top}px`,
    width: `${r.width}px`,
    height: `${r.height}px`,
  });
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
    $("camera-input").click(); // the phone's own camera app
    return;
  }
  $("camera-dialog").showModal();
  try {
    await startCamera(camDevices[camIndex]?.deviceId);
    camDevices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    $("camera-flip").hidden = camDevices.length < 2;
  } catch (err) {
    $("camera-dialog").close();
    if (window.matchMedia("(pointer: coarse)").matches) {
      $("camera-input").click();
    } else {
      setStatus(
        err?.name === "NotAllowedError"
          ? "Camera permission was refused."
          : "No camera available. Upload or paste an image instead.",
        true,
      );
    }
  }
}

function capture() {
  const el = $("cam-video");
  if (!el.videoWidth) return;
  const r = guideRect();
  const s = el.videoWidth / r.W;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(r.width * s);
  canvas.height = Math.round(r.height * s);
  canvas
    .getContext("2d")
    .drawImage(el, r.left * s, r.top * s, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
  $("camera-dialog").close();
  useSource(canvas);
}

// ---------------------------------------------------------------- notation (as in PsalmScroller)

// Keep only the chosen voices: filter %%staves and drop the other V: lines and their music.
function filterABCByVoices(abc, keep) {
  const lines = abc.split("\n");
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const staves = line.match(/^%%staves\s*\[(.*?)\]/);
    const voice = line.match(/^V:\s*([SATB])\b/);
    if (staves) {
      const groups = staves[1]
        .split(/\)\s*\(/)
        .map((g) =>
          g
            .replace(/[()]/g, "")
            .split(/\s+/)
            .filter((v) => v && keep.includes(v)),
        )
        .filter((g) => g.length);
      if (groups.length) out.push(`%%staves [${groups.map((g) => `(${g.join(" ")})`).join(" ")}]`);
    } else if (voice) {
      if (keep.includes(voice[1])) {
        out.push(line);
        if (i + 1 < lines.length) out.push(lines[++i]);
      } else {
        i++; // skip the voice's music line too
      }
    } else {
      out.push(line);
    }
  }
  return out.join("\n");
}

function render(withAudio = true) {
  const abc = $("abc").value;
  const has = /^V:/m.test(abc) || /^K:/m.test(abc);
  $("empty").hidden = has;
  if (!has) {
    $("paper").innerHTML = "";
    return;
  }
  let visualObj;
  try {
    visualObj = ABCJS.renderAbc("paper", filterABCByVoices(abc, voices), {
      responsive: "resize",
      visualTranspose: transpose,
      add_classes: true,
    })[0];
  } catch (err) {
    console.warn("Render problem:", err);
    return;
  }
  if (!withAudio) return;
  if (!ABCJS.synth.supportsAudio()) {
    $("audio").textContent = "Audio playback isn't supported in this browser.";
    return;
  }
  // A fresh controller each time (as PsalmScroller does): once a tune has played, setTune on the
  // old one keeps playing the old audio, ignoring transpose and voice changes.
  synthControl?.destroy();
  synthControl = new ABCJS.synth.SynthController();
  synthControl.load("#audio", null, { displayPlay: true, displayProgress: true, displayRestart: true });
  synthControl.setTune(visualObj, false, { chordsOff: true, midiTranspose: transpose }).catch((err) => {
    console.warn("Audio problem:", err);
  });
}

let renderTimer = null;
function renderSoon() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(() => render(true), 300);
}

// While the model writes: redraw at most every 600 ms, without rebuilding audio.
let lastLive = 0;
function renderThrottled() {
  const now = performance.now();
  if (now - lastLive > 600) {
    lastLive = now;
    render(false);
  }
}

function updateTranspose(value) {
  transpose = value;
  $("transpose-display").textContent = value > 0 ? `+${value}` : String(value);
  render(true);
}

function toggleVoice(v) {
  const next = voices.includes(v) ? voices.replace(v, "") : voices + v;
  voices = ["S", "A", "T", "B"].filter((x) => next.includes(x)).join("") || "SATB";
  for (const x of "SATB") {
    $(`voice-${x}`).classList.toggle("active", voices.includes(x));
    $(`voice-${x}`).setAttribute("aria-pressed", voices.includes(x));
  }
  render(true);
}

async function copyABC() {
  const button = $("copy-button");
  try {
    await navigator.clipboard.writeText($("abc").value);
    button.textContent = "Copied";
  } catch {
    $("abc").select();
    button.textContent = "Press Ctrl+C";
  }
  setTimeout(() => (button.textContent = "Copy"), 1500);
}

function downloadABC() {
  const abc = $("abc").value;
  const key = (abc.match(/^K:\s*(\S+)/m) || [])[1] || "";
  const composer = ((abc.match(/^C:\s*(.+)$/m) || [])[1] || "chant").replace(/.*\s/, "");
  const name = (composer + key).replace(/[^\w#-]/g, "") || "chant";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([abc], { type: "text/vnd.abc" }));
  a.download = `${name}.abc`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------------------------------------------------------------- wiring

$("load-button").addEventListener("click", loadModel);
$("transcribe-button").addEventListener("click", transcribe);
$("stop-button").addEventListener("click", () => worker.postMessage({ type: "stop" }));
$("file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) useImage(e.target.files[0]);
  e.target.value = ""; // choosing the same file again still fires
});
$("camera-input").addEventListener("change", (e) => {
  if (e.target.files[0]) useImage(e.target.files[0]);
  e.target.value = "";
});
$("camera-button").addEventListener("click", openCamera);
$("camera-shoot").addEventListener("click", capture);
$("camera-close").addEventListener("click", () => $("camera-dialog").close());
$("camera-dialog").addEventListener("close", stopCamera);
$("camera-flip").addEventListener("click", () => {
  camIndex = (camIndex + 1) % camDevices.length;
  startCamera(camDevices[camIndex].deviceId).catch((err) => console.warn(err));
});
$("guide-shape").addEventListener("input", layoutGuide);
window.addEventListener("resize", () => {
  if ($("camera-dialog").open) layoutStage();
});
$("paste-button").addEventListener("click", pasteFromClipboard);
$("abc").addEventListener("input", renderSoon);
$("transpose-down").addEventListener("click", () => updateTranspose(transpose - 1));
$("transpose-up").addEventListener("click", () => updateTranspose(transpose + 1));
for (const v of "SATB") $(`voice-${v}`).addEventListener("click", () => toggleVoice(v));
$("copy-button").addEventListener("click", copyABC);
$("download-button").addEventListener("click", downloadABC);

document.addEventListener("paste", (e) => {
  if (e.target === $("abc")) return; // pasting text into the editor
  const file = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith("image/"));
  if (file) {
    e.preventDefault();
    useImage(file);
  }
});

// Drag and drop anywhere on the page.
let dragDepth = 0;
document.addEventListener("dragenter", (e) => {
  if (![...e.dataTransfer.types].includes("Files")) return;
  dragDepth++;
  $("image-card").classList.add("dragging");
});
document.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    $("image-card").classList.remove("dragging");
  }
});
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("image-card").classList.remove("dragging");
  const file = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
  if (file) useImage(file);
});

updateUI();
