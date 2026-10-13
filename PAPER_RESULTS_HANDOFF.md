# Paper results handoff — RTX 4070 Laptop evaluation

Prepared 13 October 2026. This file contains the measured numbers, takeaways and draft text for the paper/reviewer response. Values are extracted from the completed analysis, not new experiments. The agreed trend-testing scope is complete. Numerical findings and wording are ready for author review; this does not mean accuracy, hardware speedup or headset end-to-end performance was established.

**Put these findings in the main paper**

1. Keep both networks loaded: persistent local means are 0.260 s (Total3D), 2.614 s (Implicit3D), and 2.507 s (SceneFormer), versus cold means of 6.016, 8.973 and 9.412 s.
2. Describe the actual serving policy: one loaded 2D detector and one loaded HSU model per server process; requests wait and execute sequentially through that shared pair, including mesh generation and gzip serialization.
3. Show that the LDIF models are constrained mainly by HSU forward plus mesh generation: approximately 82% of their separately profiled pipeline.
4. Show the finite queueing trend: at eight simultaneous clients, median response latency is 8.09 s for Implicit3D and 9.94 s for SceneFormer. At one arrival per second, their finite-trial medians are 17.37 and 18.20 s; at one arrival every three seconds, 2.43 and 2.71 s.
5. Report real payload sizes and compression cost: mean gzip-1 payloads are 0.53 MiB for Total3D and 1.86 MiB for both LDIF models. Level 9 provides less than 1% additional size reduction over level 6 for LDIF outputs, while adding approximately one second of CPU compression.
6. Report the denominator: 94/100 distinct images produced structurally valid finite scenes in each main suite. The other six had no detections. This is operational output validity, not reconstruction accuracy.

**Methods paragraph — draft for the paper**

> We evaluated Total3D, Implicit3D and SceneFormer on an HP ZBook Studio 16-inch G10 laptop with an Intel Core i7-13700H and an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), using WSL2 Ubuntu 24.04, Python 3.8.6, PyTorch 2.0.1, torchvision 0.15.2 and CUDA 11.8. The laptop was connected to mains power in performance mode, with competing GPU workloads closed. The same 100 RGB images and camera-intrinsic matrices and the same model checkpoints were used for local, fresh-process and loopback HTTP conditions. Each condition used three measured passes; persistent main-suite conditions used three warm-up requests excluded from the reported statistics. The server initialized one Faster R-CNN ResNet50/FPN detector and one HSU network and serialized access to their shared inference pipeline. The detector confidence threshold was retained at 0.7, and output was the existing mesh representation encoded as JSON with gzip level 1. Timing statistics describe valid scene outputs; no-detection outcomes were retained in the attempted-request denominator and reported separately.

Hardware/reproducibility details: WSL exposes 20 logical processors; kernel `6.6.87.2-microsoft-standard-WSL2`; queried Windows NVIDIA driver `617.42`; GPU compute capability 8.9; cuDNN 8.7. GPU TGP and exact OEM cooling preset were not recorded. Do not substitute a desktop RTX 4070 or infer TGP from an idle power reading. All model weights and supplied priors were retained; there was no retraining. SceneFormer demo inference on PyTorch 2.0.1 uses math SDP attention because the optimized path produced non-finite predictions on six inputs. This implementation detail should appear in reproducibility notes. cuDNN benchmarking was enabled and deterministic mode was not enabled.

**Timer definitions and experimental counts — keep with the tables**

| Condition | Timer boundary | Inputs/repetitions | Warm-up |
|---|---|---|---|
| Persistent local | Decode already-read RGB/camera bytes, unchanged detection/HSU/mesh pipeline, produce gzip JSON; CUDA synchronized before and after. File reads and output validation excluded. | 100 distinct inputs × 3 measured passes/model | 3 requests |
| Fresh process / cold | Worker initialization + loading both networks + first decoded-input-to-gzip pipeline. Input file reads, interpreter/runner work before worker entry, validation and process exit excluded from primary time. Whole subprocess wall time separately recorded. | 100 inputs × 3 fresh workers/input/model; workers run sequentially | None |
| HTTP | Client request opening through receiving complete response bytes over WSL loopback. Multipart construction/input reads and client decompression/JSON parsing excluded. Single server process and model pair. | 100 inputs × 3 measured passes/model | 3 requests |
| Profile | Separate local run with CUDA synchronization at stage boundaries; instrumented timings are diagnostic. | 100 inputs × 3 passes/model | 3 requests |
| Common burst trend | Barrier-released clients; next burst waits for all responses. HTTP latency includes waiting for the one model pair. | Same 8 inputs × 1 pass at each of 1/2/4/8 clients/model | 1 request before the pilot |
| Common interval trend | 20 inputs dispatched independently of completion at 0.5/1/2/3 s intervals; completion measured over each finite trial. | Repetition 0 for all models: 20 inputs × 1 pass/interval | Total3D/Implicit3D: 3; shortened SceneFormer: 1 |
| Payload | Offline recompression of identical saved JSON at gzip 1/6/9; also time decompression and JSON parsing. | 4 saved scenes/model × 10 repetitions/level | None specified |

Three passes are repetitions of the same scenes, not 300 independent scenes. Burst levels and interval levels were run in a fixed order. The trend comparisons match inputs and selected repetitions, but interval warm-up counts and collection times differ; describe them as exploratory finite trends, not randomized or counterbalanced capacity estimates. Cold workers do not flush OS caches. HTTP is localhost/WSL loopback, not Quest/Wi-Fi/Unity end-to-end latency.

**Main results table — mean, median and P95 latency**

Times are seconds. P95 uses the nearest-rank statistic across valid request trials. Every row has 282 valid responses from 94 distinct scenes, 18 no-detection responses from six scenes, and zero other invalid outputs. Local/cold/HTTP input and checkpoint hashes were verified identical per model. HTTP rows use the final serialized-serving runs.

| Model | Condition | Mean (s) | Median (s) | P95 (s) | Valid / attempted |
|---|---|---:|---:|---:|---:|
| Total3D | local | 0.260 | 0.239 | 0.352 | 282/300 |
| Total3D | cold | 6.016 | 6.022 | 6.770 | 282/300 |
| Total3D | http | 0.234 | 0.211 | 0.306 | 282/300 |
| Implicit3D | local | 2.614 | 2.653 | 4.290 | 282/300 |
| Implicit3D | cold | 8.973 | 8.806 | 11.841 | 282/300 |
| Implicit3D | http | 2.311 | 2.359 | 3.664 | 282/300 |
| 3DSceneFormer | local | 2.507 | 2.501 | 4.011 | 282/300 |
| 3DSceneFormer | cold | 9.412 | 9.372 | 11.140 | 282/300 |
| 3DSceneFormer | http | 2.343 | 2.377 | 3.687 | 282/300 |

Suggested main-paper interpretation:

> Retaining both networks between requests substantially reduced response latency. Mean persistent local pipeline times were 0.260 s for Total3D, 2.614 s for Implicit3D and 2.507 s for SceneFormer, whereas fresh-process response times were 6.016, 8.973 and 9.412 s, respectively. The resulting gap includes both initialization/loading and first-inference effects. Loopback HTTP means were 0.234, 2.311 and 2.343 s under single-pair serialized serving. Because local and HTTP measurements were collected in separate runs with differing warm-up/cache and performance state, their differences cannot be interpreted as isolated transport overhead.

Do not describe HTTP as faster because of distribution, negative networking overhead, or statistically superior to local inference. No physical-network measurement was made. No historical GPU timing baseline was supplied, so these are measurements of this laptop/software configuration rather than an isolated GPU-generation speedup.

**Loading/first-inference table — appendix or deployment paragraph**

| Model | Mean worker initialization (s) | Mean first pipeline (s) | Primary cold mean (s) | Whole subprocess wall mean (s) | Cold/local mean ratio |
|---|---:|---:|---:|---:|---:|
| Total3D | 3.747 | 2.269 | 6.016 | 6.514 | 23.10× |
| Implicit3D | 4.962 | 4.011 | 8.973 | 9.756 | 3.43× |
| 3DSceneFormer | 5.513 | 3.899 | 9.412 | 10.306 | 3.75× |

Initialization is the measured worker setup/model-load path, including framework setup; it is not pure disk checkpoint-read time. First inference is also slower than warmed inference, so attributing the entire cold/local difference to loading would be incorrect. The ratios describe the two measured deployment conditions, not hardware speedups.

**Profiling table — identify the bottleneck**

These are mean stage times from the separate synchronized diagnostic. Do not mix instrumented totals with headline latency. Output postprocessing includes geometry conversion, JSON and gzip; it is not pure compression.

| Model | Detection + preparation (ms) | HSU forward + mesh (ms) | Output + JSON + gzip (ms) | HSU/mesh share | Valid / attempted |
|---|---:|---:|---:|---:|---:|
| Total3D | 94.0 | 91.1 | 83.0 | 34.0% | 285/300 |
| Implicit3D | 106.0 | 2072.6 | 337.0 | 82.4% | 282/300 |
| 3DSceneFormer | 104.8 | 2059.1 | 352.2 | 81.8% | 282/300 |

> For Implicit3D and SceneFormer, the combined HSU-forward and mesh-generation stage accounted for 82.4% and 81.8% of the instrumented pipeline, respectively. Their output-postprocessing stages contributed approximately 13–14%. Total3D’s faster pipeline had more evenly balanced detection, forward/mesh and output costs. These measurements identify the combined HSU/mesh stage as the dominant optimization target for the LDIF models, without separating neural computation from mesh extraction.

Total3D profiling reconstructed one object on sample 96 in all three repetitions, while its main local/cold/HTTP runs returned no detections on that input. The cause is not established. This explains 285/300 valid profiling outputs versus 282/300 in main suites; do not quietly report the profiling success denominator as identical.

Optional appendix memory figures: maximum PyTorch-allocated and PyTorch-reserved memory across valid profiled requests. These include the loaded PyTorch model pair and allocations visible to that process but exclude external LDIF processes, other CUDA contexts and total system GPU memory.

| Model | Max PyTorch allocated (MiB) | Max PyTorch reserved (MiB) |
|---|---:|---:|
| Total3D | 1697.7 | 4120.0 |
| Implicit3D | 958.6 | 1380.0 |
| 3DSceneFormer | 939.5 | 1216.0 |

Do not use this table to claim total GPU memory requirements, memory per concurrent user, or capacity for additional model copies.

**Burst table — enough to show the serialized waiting trend**

Median valid HTTP response times in seconds; common eight-image pilot at each client level, seven valid responses and one no-detection response. One pass/model/level. The same detector/HSU pair serves all requests sequentially; these are concurrent clients, not concurrent model replicas.

| Model | 1 client | 2 clients | 4 clients | 8 clients |
|---|---:|---:|---:|---:|
| Total3D | 0.287 | 0.262 | 0.448 | 0.851 |
| Implicit3D | 2.486 | 3.227 | 5.090 | 8.090 |
| 3DSceneFormer | 2.479 | 3.288 | 5.646 | 9.938 |

> Simultaneous requests were serialized through one persistent detector/HSU pair. In the common eight-image burst experiment, median valid-response latency for Implicit3D increased from 2.486 s at one client to 8.090 s at eight clients; SceneFormer increased from 2.479 s to 9.938 s. These observations illustrate waiting-time growth under shared sequential inference, rather than parallel model replication or an eight-user capacity guarantee.

The Total3D dip from one to two clients is small-sample/ordered-run variation, not evidence of parallel model acceleration. Larger completed Total3D/Implicit3D burst runs exist, but do not combine their repetitions with the deliberately shortened SceneFormer sweep to claim an equal full-size comparison. The balanced pilot is the main comparative trend evidence.

**Arrival-rate table — show accumulation of waiting work**

Median valid HTTP response times in seconds; twenty common images and the first repetition at each dispatch interval. Each trial has 18 valid scenes and two no-detection outcomes. Warm-up counts differ as described in the methods table.

| Model | 0.5 s interval | 1 s interval | 2 s interval | 3 s interval |
|---|---:|---:|---:|---:|
| Total3D | 0.745 | 0.257 | 0.318 | 0.471 |
| Implicit3D | 22.058 | 17.367 | 7.228 | 2.431 |
| 3DSceneFormer | 22.783 | 18.203 | 8.564 | 2.710 |

> When requests arrived more quickly than the shared pair could process them, response latency increased substantially. At one-second dispatch intervals, median valid responses were 17.367 s for Implicit3D and 18.203 s for SceneFormer. At three-second intervals, the corresponding medians were 2.431 and 2.710 s for these finite trials, near isolated-request latency. Thus, maintaining a persistent pair removes reload overhead but does not remove the waiting caused by arrival rates exceeding its processing rate.

Finite-trial valid-response rates (valid scenes divided by full trial duration, including drain time):

| Model | 0.5 s interval (valid/s) | 1 s (valid/s) | 2 s (valid/s) | 3 s (valid/s) |
|---|---:|---:|---:|---:|
| Total3D | 1.697 | 0.943 | 0.473 | 0.314 |
| Implicit3D | 0.405 | 0.413 | 0.413 | 0.305 |
| 3DSceneFormer | 0.394 | 0.400 | 0.394 | 0.304 |

The roughly 0.39–0.41 valid scenes/s for the two LDIF models at faster arrival rates is finite-trial goodput, not steady-state capacity. Do not turn three-second dispatch intervals into a universal minimum safe interval, a maximum number of users, or proof of indefinite queue stability. Trials were short, ordered, and input-dependent. No additional exhaustive sweeps are required to demonstrate the observed trend.

**Payload sizes — main paper communication-cost table**

Means over valid outputs in the full serialized HTTP suites. MiB means 2^20 bytes; do not relabel as decimal MB. Byte reduction is computed from the ratio of aggregate mean sizes.

| Model | Mean JSON (MiB) | Mean gzip-1 (MiB) | Byte reduction |
|---|---:|---:|---:|
| Total3D | 1.256 | 0.526 | 58.1% |
| Implicit3D | 6.774 | 1.863 | 72.5% |
| 3DSceneFormer | 6.774 | 1.863 | 72.5% |

> The existing gzip-1 mesh representation reduced mean payload size from 1.256 to 0.526 MiB for Total3D and from approximately 6.774 to 1.863 MiB for the LDIF models, corresponding to reductions of 58.1% and approximately 72.5%. These measurements quantify the current representation’s byte volume; they do not establish superiority to binary meshes, Gaussian splatting or another representation, nor measure transmission over the physical headset link.

**Compression trade-off — appendix and short main-paper takeaway**

Mean CPU compression times and resulting sizes for exactly the same saved JSON, four valid smoke payloads/model and ten repetitions/payload/level. These are a different sample set from the full HTTP payload-size table, so sizes need not match that table. Production remained gzip level 1.

| Model | Gzip level | Compressed size (MiB) | Compression (ms) | Decompression (ms) | JSON parsing (ms) |
|---|---:|---:|---:|---:|---:|
| Total3D | 1 | 0.497 | 10.6 | 5.1 | 11.6 |
| Total3D | 6 | 0.461 | 74.7 | 4.7 | 11.7 |
| Total3D | 9 | 0.456 | 243.9 | 4.7 | 11.7 |
| Implicit3D | 1 | 1.771 | 41.1 | 21.4 | 77.7 |
| Implicit3D | 6 | 1.433 | 263.1 | 19.3 | 77.0 |
| Implicit3D | 9 | 1.422 | 1317.1 | 19.6 | 76.5 |
| 3DSceneFormer | 1 | 1.770 | 40.5 | 21.3 | 76.4 |
| 3DSceneFormer | 6 | 1.433 | 262.2 | 19.4 | 75.7 |
| 3DSceneFormer | 9 | 1.422 | 1285.6 | 19.7 | 75.2 |

| Model | Additional bytes saved: level 6 vs 1 | Added compression: level 6 vs 1 (ms) | Additional bytes saved: level 9 vs 6 | Added compression: level 9 vs 6 (ms) |
|---|---:|---:|---:|---:|
| Total3D | 7.3% | 64.1 | 1.2% | 169.2 |
| Implicit3D | 19.1% | 222.0 | 0.8% | 1054.0 |
| 3DSceneFormer | 19.1% | 221.7 | 0.8% | 1023.3 |

> Higher compression levels trade byte savings for additional CPU time. For the LDIF payloads, level 6 reduced size by a further 19.1% relative to level 1 while adding about 222 ms of compression. Level 9 saved only another 0.8% relative to level 6 while adding about one second. These observations support retaining gzip level 1 for latency-sensitive use of the current representation, while leaving compression choice dependent on deployment requirements.

At gzip level 1, offline decompression plus JSON parsing averaged about 17 ms for Total3D and 98–99 ms for the LDIF payloads. Those CPU measurements were made on the laptop and are excluded from HTTP latency; they do not measure Quest/Unity processing, rendering or energy.

**Output validity and limitations — text that must accompany the results**

> Each main evaluation condition generated structurally valid finite scene outputs for 94 of the 100 distinct inputs, giving 282 valid and 18 no-detection responses over three passes. The six no-detection inputs were retained in the denominator. This operational output rate is not a measure of 2D detection or 3D reconstruction accuracy. No labelled ground truth was available to quantify detection mAP, layout IoU, pose error or mesh distance. The measurements also exclude the physical headset/network/rendering path and do not establish client energy savings, a historical GPU speedup or sustained multi-user capacity.

Specific qualification notes for the author:

- No-detection input IDs: 89, 90, 96, 98, 99, 100. Same detector and threshold across models.
- Total3D profiling-only sample-96 discrepancy: 285 valid outputs rather than 282; unexplained. Profile data are diagnostic and remain separate.
- Original SceneFormer non-finite outputs on 53, 68, 70, 73, 84, 97 were resolved by the math attention backend. Original failed records are preserved; final table rows use the corrected implementation.
- The initial unsynchronized burst attempt timed out and revealed a Python 3.8 timeout-classification bug. That run is excluded from successful serving claims. Final serialized comparisons had zero unexpected errors.
- The long SceneFormer burst was stopped intentionally when enough trend evidence existed. Only the completed balanced pilot enters the common burst table.
- Operational finite-geometry checks do not verify object placement, semantic correctness or domain generalization. The sample source/provenance must be confirmed before publication; no labelled Quest dataset was established here.
- Counts/repetitions are descriptive. Do not invent confidence intervals, statistical significance, independent-scene counts, a 30-user validation, a maximum supported client count or a steady-state throughput claim.

**Reviewer/section mapping**

Mappings follow the project’s REVISION_TESTS.md; reconcile them with the actual reviewer wording before submitting.

| Concern | Paper insertion | Evidence / response boundary |
|---|---|---|
| R1.2–3, R2.2, R3.2: latency and loading | Evaluation setup + latency/loading tables | Distinguish persistent serving from fresh processes; use identical inputs/weights. Avoid claiming isolated networking or hardware-generation gains. |
| R1.4: mesh-transfer cost | Payload table + compression trade-off paragraph | Actual byte volumes and CPU costs are measured. Alternatives were not compared; physical headset transfer was not measured. |
| R1.5, R3.4: concurrent users/arrivals | Deployment policy + burst/interval tables | One pair, sequential inference, waiting latency. Describe finite observed trends; no sustained-capacity or maximum-user guarantee. |
| R2.4, R3.5: reconstruction accuracy | Limitations / outstanding evaluation | These runs do not close the accuracy evidence gap. Ground truth and metric conventions are still required. |
| R2.5: Quest domain gap | Dataset provenance + limitations | No labelled Quest-domain accuracy evidence from these inputs. |
| R2.3: client energy | Claims audit + limitations | No energy measurement. Qualify/remove quantitative measured-benefit wording; do not imply this benchmark establishes it. |

**Suggested figure captions**

- Performance figure: “Operational latency and serialized-serving trends on an RTX 4070 Laptop GPU. Main latency conditions used 100 distinct input pairs and three measured passes; plotted timing statistics include valid scene responses, with no detections retained in outcome counts. Profiling is a separate synchronized diagnostic. Burst trends use a common eight-input single-pass pilot at each client level. Interval trends use the first twenty-input repetition at each dispatch interval. HTTP uses WSL loopback; finite trends do not establish sustained user capacity.”
- Compression figure: “CPU compression cost and byte volume for identical saved scene JSON recompressed at gzip levels 1, 6 and 9. Four valid smoke scenes per model were tested ten times per level. These offline CPU timings do not change the production gzip-1 format or measure physical headset transfer.”

**Exact artifacts and reproducibility handoff**

- Analysis and validated source extraction: [REPORT.md](revision_results/analysis-20261013/REPORT.md), [analysis.json](revision_results/analysis-20261013/analysis.json).
- Editable numeric tables: [latency.csv](revision_results/analysis-20261013/latency.csv), [profile.csv](revision_results/analysis-20261013/profile.csv), [payload.csv](revision_results/analysis-20261013/payload.csv), [burst.csv](revision_results/analysis-20261013/burst.csv), [interval.csv](revision_results/analysis-20261013/interval.csv).
- Paper figures: [performance-trends.pdf](revision_results/analysis-20261013/performance-trends.pdf), [compression-tradeoff.pdf](revision_results/analysis-20261013/compression-tradeoff.pdf). PNG versions are in the same folder.
- Full local/cold/profile and payload records: `revision_results/20261013-measurements/full-<model>-<suite>/`. Each contains raw records and manifests.
- Final serialized HTTP and common burst pilots: `revision_results/20261013-serialized-concurrency/full-<model>-http/` and `pilot-<model>-load/`.
- Interval sources: `full-Total3D-intervals/`, `full-Implicit3D-intervals/`, `trend-3DSceneFormer-intervals/` under the serialized directory; use repeat 0 for the common comparison.
- Manifests record commands, input/checkpoint/prior hashes, effective config, packages, hardware metadata and working diff. Ready/runtime records capture server source hash, inference policy, PID and attention backend settings for final serving runs.
- `revision_results/` is ignored by Git. This root-level handoff contains the paper numbers by itself; transfer the analysis folder and raw records separately when handing the evidence to another machine/author.

Checkpoint SHA-256 values for reproducibility (paths relative to the repository):

| Model / component | Checkpoint | SHA-256 |
|---|---|---|
| Total3D | `Total3DUnderstanding/out/pretrained_models/pretrained_model.pth` | `25938634922a0f195ef75504d1c9ce82ae54c23b3ded98e19e2495994e250fdf` |
| Shared 2D detector | `2D_Detectors/Resnet50/fine_tuned_resnet.pth` | `2011fb0ea136731de598f0b420ddcbb3aadec68e0064110431ace37c27cee446` |
| Implicit3D | `Implicit3DUnderstanding/out/total3d/20110611514267/model_best.pth` | `9968efa609b11e6ab5f571526893979f869b95a5d8287797cf7244846c3c03b5` |
| 3DSceneFormer | `ThreeDSceneFormer/out/total3d/transformer_enc_dec/24100215400096/model_best.pth` | `09c3f42e32e68df5d4d3962c73542588bef8d8f7cb512355a3752db30df8876d` |

**Final manuscript checklist**

- Insert hardware/software and exact timer definitions before comparing numbers.
- Insert the main latency table, profiling breakdown and payload table.
- Include burst/arrival plots as finite trends with their smaller sample/repetition scope.
- State the shared single-pair serialized policy; do not describe simultaneous requests as spawning model replicas.
- Report no-detection counts and condition latency statistics on valid outputs.
- Keep the profiling discrepancy and implementation changes visible in reproducibility notes.
- Qualify claims about accuracy, headset domain, physical networking, energy, historical hardware improvement and sustained capacity.
- Author-review the numerical findings and final wording against the actual manuscript/reviewer comments; no further runs are needed for the agreed operational trend demonstration.
