ONNX Runtime Web 1.30.0 (MIT licence, Copyright (c) Microsoft Corporation), copied from the
`onnxruntime-web` npm package: `dist/ort.wasm.bundle.min.mjs` and `dist/ort-wasm-simd-threaded.wasm`.
Used by the offline captions recogniser (js/offline/). Only the single-threaded WebAssembly CPU
backend is used, because GitHub Pages cannot serve the headers that threads need.
