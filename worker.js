// Runs the PsalmOMR model (LFM2.5-VL-450M fine-tune, ONNX) with Transformers.js, off the main thread.
import {
  AutoModelForImageTextToText,
  AutoProcessor,
  env,
  InterruptableStoppingCriteria,
  RawImage,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";

const MODEL_ID = "yclicc/PsalmOMR-LFM2.5-VL-450M-GGUF";
// The model was trained on this one instruction, word for word.
const PROMPT =
  "Transcribe this Anglican psalm chant into ABC notation (four voices S A T B on two staves). Output only the ABC.";
// Language model and output head ggml Q4_0 (what training simulated); vision and embeddings fp16
// (onnxruntime-web has no kernel for 4-bit embedding lookups).
const DTYPE = { vision_encoder: "fp16", embed_tokens: "fp16", decoder_model_merged: "q4" };
// Weights sit in separate .onnx_data files. config.json says so too, but saying it here means an
// older cached config.json can't break loading.
const MODEL_OPTIONS = { dtype: DTYPE, use_external_data_format: true };
// On WebAssembly, ONNX Runtime's 4-bit MatMulNBits unpacks its weights on every call. onnx_wasm/
// holds the same Q4_0 weights as DequantizeLinear + MatMul, which this option makes it expand to
// float once at load instead: ~8x faster per token, ~1.5 GB of memory.
const WASM_OPTIONS = {
  subfolder: "onnx_wasm",
  session_options: { extra: { session: { disable_quant_qdq: "1" } } },
};

function optionsFor(device) {
  return { ...MODEL_OPTIONS, ...(device === "wasm" ? WASM_OPTIONS : {}), device };
}

// Without WebGPU, ONNX Runtime runs on WebAssembly: one thread unless the page is cross-origin
// isolated (coi-serviceworker.js), then as many as the CPU has (ONNX Runtime's default is half).
const THREADS = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
env.backends.onnx.wasm.numThreads = THREADS;

let processor = null;
let model = null;
let device = null;
const stopping = new InterruptableStoppingCriteria();

async function pickDevice() {
  try {
    if (navigator.gpu && (await navigator.gpu.requestAdapter())) return "webgpu";
  } catch {
    // fall through
  }
  return "wasm";
}

async function load() {
  const files = {};
  const progress_callback = (p) => {
    if (p.status === "progress" && p.total) {
      files[p.file] = { loaded: p.loaded, total: p.total };
      let loaded = 0;
      let total = 0;
      for (const f of Object.values(files)) {
        loaded += f.loaded;
        total += f.total;
      }
      self.postMessage({ type: "progress", loaded, total });
    }
  };
  device = await pickDevice();
  processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback });
  try {
    model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, { ...optionsFor(device), progress_callback });
  } catch (err) {
    if (device !== "webgpu") throw err;
    console.warn("WebGPU failed, falling back to WebAssembly:", err);
    device = "wasm";
    model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, { ...optionsFor(device), progress_callback });
  }
  self.postMessage({
    type: "ready",
    device,
    threads: THREADS,
    // Why only one thread, if so: service workers (coi-serviceworker.js) need https or localhost.
    whySingle: self.crossOriginIsolated ? null : self.isSecureContext ? "not-isolated" : "not-secure",
  });
}

async function transcribe({ data, width, height }) {
  const image = new RawImage(new Uint8ClampedArray(data), width, height, 4).rgb();
  const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: PROMPT }] }];
  const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(image, prompt, { add_special_tokens: false });
  const start = performance.now();
  let tokens = 0;
  let first = null; // time to the first token: image encoding + prompt
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => self.postMessage({ type: "token", text }),
    token_callback_function: () => {
      tokens++;
      first ??= performance.now();
    },
  });
  stopping.reset();
  const ids = await model.generate({
    ...inputs,
    do_sample: false,
    max_new_tokens: 1024,
    streamer,
    stopping_criteria: stopping,
  });
  const n = inputs.input_ids.dims.at(-1);
  const text = processor.batch_decode(ids.slice(null, [n, null]), { skip_special_tokens: true })[0];
  const end = performance.now();
  self.postMessage({
    type: "done",
    text,
    seconds: (end - start) / 1000,
    firstSeconds: ((first ?? end) - start) / 1000,
    tokensPerSecond: tokens > 1 ? ((tokens - 1) * 1000) / (end - first) : 0,
  });
}

self.addEventListener("message", async ({ data: msg }) => {
  try {
    if (msg.type === "load") await load();
    else if (msg.type === "transcribe") await transcribe(msg);
    else if (msg.type === "stop") stopping.interrupt();
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message ?? err) });
  }
});
