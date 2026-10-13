# Revision measurement protocol

For installation and initial validation, follow [SETUP_4070.md](SETUP_4070.md). Use `bash run_4070.sh` before each Python script name in the commands below so the Conda toolkit and WSL driver are selected. No laptop inference or new timings have been validated yet.

## Configuration and implementation changes

- The author confirmed SceneFormer checkpoint `out/total3d/transformer_enc_dec/24100215400096/model_best.pth` and supplied Implicit3D checkpoint `out/total3d/20110611514267/model_best.pth`. The corresponding default YAML weight paths now select those files. Other model settings were retained.
- Total3D continues to select `out/pretrained_models/pretrained_model.pth`.
- The author-supplied `layout_avg_file.pkl` and `size_avg_category.pkl` were copied byte-for-byte from the repository root into each model's `data/sunrgbd/preprocessed/` directory. These configuration classes read the priors during inference to decode predictions. The source copies remain at the root.
- All three checkpoint loaders now resolve relative weights against their model package, accept absolute paths, and raise for missing demo checkpoints instead of continuing with random weights. Training-mode handling was retained.
- The first smoke attempt exposed import-time compilation of the ChamferDistance metric, with a missing build directory. Its three wrappers now load/build the extension on first metric use, use absolute source paths and create the build directory. Inference does not need to evaluate this metric. CPU checks cover the deferred-loading logic; actual metric compilation and GPU inference remain unverified.
- The runner records checkpoint hashes and audits the loaded network state, including inference buffers. Only BatchNorm `num_batches_tracked` counters are exempt from coverage, since eval-mode inference does not use them.
- Each run writes an effective YAML with absolute checkpoint paths, run-specific logs, GPU enabled and finetuning/checkpoint loading enabled. Automatic resume is disabled to prevent a log-directory checkpoint from silently overriding the selected weights.
- The test runner selects **ResNet50 only**. The current API's MobileNet/DETR options instantiate a Faster R-CNN ResNet50 and construct the wrong checkpoint filenames for those alternatives. They cannot be treated as working alternative detectors. No detector architecture was changed.
- No task queue, serializer, model architecture, detector threshold or mesh resolution was changed. The serialized output remains the existing gzip level-1 JSON.
- `revision_results/` is ignored by Git. Existing local setup changes were preserved; code edits have not been committed or pushed.

## Measurements to run after smoke succeeds

Use the same model and sample limit across local/cold/HTTP conditions. Defaults are all 100 images, three measured passes, and three explicitly logged warm-up requests for persistent modes. These are proposed repetition counts recorded by the script, not a claim that three passes establish statistical stability. Run conditions sequentially with no other GPU workload. Do not treat repeated predictions of one image as independent scenes.

```bash
bash run_4070.sh run_revision_tests.py --suite local --model Total3D
bash run_4070.sh run_revision_tests.py --suite cold --model Total3D
bash run_4070.sh run_revision_tests.py --suite http --model Total3D
bash run_4070.sh run_revision_tests.py --suite profile --model Total3D
bash run_4070.sh run_revision_tests.py --suite load --model Total3D --concurrency 1 2 4 8
bash run_4070.sh run_revision_tests.py --suite intervals --model Total3D --limit 20 --intervals 0.5 1 2 3
```

Repeat with the other two HSU models once their smoke tests pass. `--limit 5 --repeats 1` gives a pilot, which must not be reported as the complete experiment. Fresh-process tests can take substantially longer because they initialise both networks for every image. They do not flush OS filesystem caches.

| Suite | Measurement boundary and purpose |
|---|---|
| `local` | Persistent models; synchronise CUDA, decode the already-read RGB and camera bytes, run the unchanged pipeline through gzip-JSON output, synchronise CUDA. File reads and response validation are outside the timer. This is the approved persistent non-distributed baseline. |
| `cold` | Fresh worker per image; primary time spans worker initialisation, model loading and the same decoded-input-to-gzip pipeline. Interpreter/runner imports before the worker entry, input file reads and worker exit are excluded. Whole subprocess wall time is separately recorded and includes startup, validation, logging and exit. It is not a claim of a disk-cold cache. |
| `http` | Persistent server; client time spans opening the HTTP request through receiving the complete response bytes. Input reads/multipart construction and client decompression/JSON parsing are outside this timer. Loopback includes Flask/request handling but not Quest capture, Wi-Fi or Unity rendering. |
| `profile` | Separate local diagnostic pass synchronising CUDA at the existing stage timer boundaries. Stages are detection **and input preparation**, HSU forward **including mesh generation**, and output postprocessing **including JSON and gzip**. Do not label the last stage pure compression or use this instrumented total as the headline latency. |
| `load` | Barrier-released request bursts at specified concurrency; the next burst waits for all clients. Records actual and requested burst sizes, including a smaller final batch. No queue was added. This is a finite burst test, not a sustained user-capacity model. |
| `intervals` | A finite trial of inputs dispatched at planned intervals independently of response completion; dispatch lag is recorded. Defaults to 20 distinct images per trial and three repetitions. Response rate is measured over the full finite trial; it is not steady-state throughput or a 30-user validation. |
| `payload` | Offline CPU recompression at gzip levels 1, 6 and 9 of exactly the same saved JSON bytes. Records byte counts and compression/decompression/parsing time. Does not change the production output format. |

For payload tests, pass an actual completed smoke-run directory:

```bash
bash run_4070.sh run_revision_tests.py --suite payload --payload-dir revision_results/<completed-smoke-directory> --repeats 10
```

Raw `requests.jsonl` includes every attempted request, scene ID, repetition, failures, timing, and valid output sizes/counts/hashes. `summary.json` reports success-conditional latency alongside attempted/success/failed counts. Warm-up requests are marked and excluded from summaries. A timeout stops subsequent batches to avoid contaminating them with unfinished GPU work. Partial results and failure logs remain available. Only the spawned server is terminated; the existing LDIF subprocess/temp-file lifecycle remains part of the implementation under test and should be checked after crashes.

The manifest records the Git revision, working diff, configuration, checkpoint/input hashes, Python packages and available hardware metadata. Local profiling records PyTorch peak allocated/reserved memory; this does not include external LDIF processes or total system GPU memory. Results require author review before becoming paper findings. Do not derive network cost solely by subtracting independent local and HTTP means without examining paired inputs, variation and different timing boundaries.

## Review issues and scope

| Evidence needed | Current action/status | Defensible interpretation |
|---|---|---|
| New-hardware latency and loading overhead (R1.2–3, R2.2, R3.2) | Approved same-input/checkpoint/output comparison implemented; awaiting GPU runs. | Quantify loading and serving costs; retain fresh-process behaviour as a separately labelled deployment condition. |
| Mesh-transfer cost (R1.4) | Measure actual JSON/gzip bytes and CPU costs. | Judge whether this representation is adequate for occasional reconstruction updates. Gaussian splatting/binary streaming is not automatically required to evaluate an explicit-mesh integration. Do not claim superiority to untested alternatives. |
| Burst/arrival behaviour (R1.5, R3.4) | Existing Flask server inspected; load/interval tests implemented. | No application-level queue is evident in the inspected entry point. LDIF reuses per-model temporary paths (`ldif.txt`, `grid.grd`), creating a possible concurrency race. Structurally valid responses do not rule out mixed-request geometry. Decide on a bounded fix or a documented concurrency limit after runtime evidence; no queue redesign is approved. |
| 2D/3D reconstruction accuracy (R2.4, R3.5) | Ground-truth test files absent. | Unlabelled samples support operational checks, not mAP, layout IoU, pose error, mesh distance or GT-vs-predicted-box comparisons. A metric protocol needs the actual annotation conventions. |
| Quest domain gap (R2.5) | No identified labelled Quest evaluation set in this task. | No Quest accuracy or reconstruction-success claim from these samples. Input source/provenance must be confirmed before publication. |
| Client energy (R2.3) | No measurement added. | Do not present an expected benefit as a measured finding. Whether to add this experiment or revise claims remains an author decision. |

The reviewer response should distinguish substantive evidence gaps from suggested implementation choices. A reviewer requesting a queue or another representation does not establish that such a redesign is necessary for the paper's stated on-demand use case. The response should explain the tested operating conditions, failures and remaining limits, with author approval of any decision to decline a requested experiment.

## Verification so far

CPU-only checks cover sample selection, malformed outputs, non-finite geometry, face indices, failure denominators, warm-up exclusion, HTTP-200 error interpretation, checkpoint path handling and lazy metric-extension loading. Ten checks passed. A real loopback HTTP fixture was skipped because the Windows sandbox blocks socket connections. These checks do not validate CUDA, the installed `hsu` dependencies, model output quality or production inference. User-run smoke logs have been inspected: model startup failed, and the separate environment diagnostic confirms that PyTorch cannot currently access CUDA.
