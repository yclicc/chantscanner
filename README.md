# ChantScanner

Photograph an Anglican psalm chant and get it back as editable, playable ABC notation, entirely in
the browser. The model ([PsalmOMR LFM2.5-VL-450M](https://huggingface.co/yclicc/PsalmOMR-LFM2.5-VL-450M-GGUF),
ONNX build) runs locally with [Transformers.js](https://huggingface.co/docs/transformers.js), so
pictures are never uploaded anywhere. The result is shown and played with
[abcjs](https://www.abcjs.net/), with the same transpose and voice buttons as
[PsalmScroller](https://yclicc.github.io/psalmscroller).

Single and double chants only. The notes are usually right; the composer's name often isn't.

## Files

- `index.html`: page and styles.
- `app.js`: image input (file, camera, clipboard, drag and drop), resizing, ABC editor, notation,
  transpose, voice filter and playback.
- `worker.js`: loads the model and transcribes, in a Web Worker so the page stays responsive.
- `style.css`: styles.
- `coi-serviceworker.js`: third-party (MIT), see Speed below.

There is no build step. Libraries load from jsDelivr at pinned versions (Transformers.js 4.3.1,
abcjs 6.5.2).

## Running it

Serve the folder with any static server and open it:

```sh
python3 -m http.server 8000   # then open http://localhost:8000
```

Opening `index.html` straight from disk doesn't work: module workers need http(s). GitHub Pages
or any other static host works as is.

## Notes

- **Model download:** about 490 MB the first time (vision encoder fp16, language model and
  embeddings Q4_0), then served from the browser's cache.
- **Speed:** the model runs on the GPU through WebGPU where the browser has it (recent Chrome,
  Edge and Safari). Otherwise it falls back to WebAssembly on the CPU, which is much slower.
  Multi-threaded WebAssembly needs the page to be cross-origin isolated (COOP/COEP headers),
  which static hosts like GitHub Pages can't set, so `coi-serviceworker.js`
  ([coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) v0.1.7, MIT) adds them from
  a service worker. The page reloads itself once on the first visit to switch it on. The status
  line shows how many threads the CPU fallback uses, and why if it's only one. Service workers
  only run on https or `localhost`, so a phone opening the laptop's `http://192.168.…` address gets
  one thread. To test on a phone, use https hosting, an https tunnel, or on Android
  `adb reverse tcp:8000 tcp:8000` and then `http://localhost:8000`.
- **Image sizing:** each picture is scaled so its longest edge is at most 1024 px, then to the
  model's image grid (sides multiples of 32 px, at most 256 tokens of 32×32 px), the size the model
  was trained on. Shrinking happens in halving steps so thin staff lines survive.
- **Model license:** the model is a fine-tune of Liquid AI's LFM2.5-VL-450M and is used under the
  [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE). This
  site doesn't redistribute the weights; it downloads them from Hugging Face.
