# RTX 4070 Laptop: WSL Ubuntu 24.04

This setup supports a fresh `hsu` environment. On 13 October 2026, the RTX 4070 Laptop passed GPU/NMS checks, both LDIF builds and grid tests, and inference for all three models on sample folders 1, 26, 51 and 75. Sample 100 returns `No 2D detections` for every model, so the original five-image smoke suite still exits with a failure. Do not use the old RTX 5080 upgrade/restore scripts.

## 1. Environment

Clone this repository inside the WSL Linux filesystem, not `/mnt/c`. Install Conda in WSL first. The Windows NVIDIA driver supplies WSL GPU access: do not install an Ubuntu NVIDIA display driver.

```bash
git clone https://github.com/GijsFiten/Edge-XReal.git
cd Edge-XReal
conda env create -f environment-4070.yaml
conda activate hsu
```

`environment-4070.yaml` retains Python 3.8 and the original application's library pins. The hardware adaptations are PyTorch 2.0.1, torchvision 0.15.2, matching torchaudio 2.0.2, CUDA 11.8 runtime/toolkit, and a Conda GCC/G++ 11 compiler. The laptop repair also adds `libxcrypt=4.4.36` (provides `crypt.h` required to compile `mathutils` and `pickle5`) and `lit=16.0.6` (a missing torchtriton dependency). The complete CUDA toolkit dependency closure is pinned because `cuda-toolkit=11.8.0` alone allowed Conda to install a CUDA 13.1 compiler. `cuda-toolkit-4070.explicit.txt` records the NVIDIA 11.8 package URLs and checksums used for the repair. The original `environment.yaml` is unchanged. CUDA, `nvcc` and the compiler are installed inside Conda; no Ubuntu CUDA toolkit installation is required. Creating this environment is a new install, not an instruction to overwrite an existing `hsu`.

The environment is derived from the original complete export, including its optional development/training dependencies. The repaired laptop environment passes `pip check`, and the updated YAML passes a fresh Conda dry-run solve without CUDA 13 components. If a future installation fails, retain the error; avoid a blanket upgrade. No model retraining is part of this setup.

Official references: [PyTorch version pairing](https://pytorch.org/get-started/previous-versions/#v201), [CUDA 11.8 installation](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html), [WSL driver guidance](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## 2. Transfer the exact inference assets

Git does not contain the checkpoints, priors, sphere templates or sample images. Copy `Edge-XReal_4070_assets.tar` from the preparation machine to the laptop, then extract it in the fresh repository:

```bash
tar -xf /path/to/Edge-XReal_4070_assets.tar -C .
```

The archive contains the author-approved checkpoints, ResNet50 detector weights, two SUN-RGBD prior files, three sphere templates, and the 100 `sample_images` pairs. `inference_assets.json` lists their relative paths, sizes and SHA-256 hashes. No compiled binaries are transferred. This archive is for local transfer, not a Git commit or a public release.

## 3. Check, build and run standard inference

```bash
bash run_4070.sh setup_4070.py --build --smoke
```

The launcher uses the active Conda toolkit/compiler and, on WSL, gives WSL's driver libraries priority for that process. It does not modify shell startup files or system libraries.

The setup checks the GPU capability (8.9), CUDA tensor computation and torchvision NMS; verifies the asset hashes; copies the supplied priors into each model's expected folder; and checks each effective config. It builds only the LDIF executables used by Implicit3D and SceneFormer, with native `sm_89` code plus `compute_89` PTX. Each new binary must pass a small grid-generation test before replacing an old binary. It then runs the existing inference pipeline for all three models on sample folders 1, 26, 51, 75 and 100. The lazy Chamfer loader avoids building a training/evaluation metric during inference imports.

Logs and returned gzip-JSON scenes go under `revision_results/`. A failure stops this setup run. A successful smoke result means structurally valid output, not proven reconstruction accuracy. Do not start measurements until all three models pass and their returned scenes have been inspected.

For a quick inference check on the repaired laptop, use sample 1 explicitly via `--limit 1`:

```bash
conda activate hsu
for model in Total3D Implicit3D 3DSceneFormer; do
  bash run_4070.sh run_revision_tests.py --suite smoke --model "$model" --limit 1 || exit 1
done
```

The original five-image smoke command above still tests sample 100 and exits nonzero on its no-detection response. Do not interpret the one-image check as a complete benchmark qualification. The repair's five-image results are in `revision_results/4070-repair-smoke-Total3D-v3`, `4070-repair-smoke-Implicit3D-v2` and `4070-repair-smoke-3DSceneFormer-v2`. Each records four valid scenes and the sample-100 error. The API import in the measurement runner now loads the repository's `server.py` explicitly; the two LDIF models defer dataset-only evaluation imports; Total3D keeps indices on CPU when indexing its CPU mesh templates.

The full dataset later exposed non-finite SceneFormer predictions on six additional inputs. Its demo API now uses math SDP attention on the pinned PyTorch 2.0.1 stack. This resolved all six inputs without changing checkpoint weights or architecture; all seven targeted API regression inputs pass. Runtime metadata records the backend selection. See `REVISION_TESTS.md` for the preserved baseline and measurement rerun notes.

To start the application:

```bash
conda activate hsu
bash run_4070.sh server.py --hsu_model 3DSceneFormer --bb_model Resnet50
```

For one model later:

```bash
bash run_4070.sh run_revision_tests.py --suite smoke --model Total3D
```

## 4. Approved measurements

Use mains power and a consistent Windows/laptop performance mode. Record the laptop model, GPU power/TGP setting, cooling mode, Windows driver, WSL version and any competing GPU workload in `revision_results/laptop_conditions.txt`. The runner also captures hardware and package metadata. These are RTX 4070 **Laptop GPU** results, not desktop RTX 4070 or RTX 5080 results.

Run each model and suite sequentially:

```bash
for model in Total3D Implicit3D 3DSceneFormer; do
  for suite in local cold http profile; do
    bash run_4070.sh run_revision_tests.py --suite "$suite" --model "$model" || exit 1
  done
done
```

Local/cold/HTTP defaults use all 100 available images and three repetitions; persistent local and HTTP runs warm up first. `profile` is a separately instrumented diagnostic. The first three suites compare identical images, checkpoints and gzip-JSON output representation. Localhost HTTP is not Quest/Wi-Fi/end-to-end latency. Changes from historical timings reflect both hardware and software differences.

For burst and interval experiments, see `REVISION_TESTS.md`. At the user's request, the API now initializes one detector/HSU pair and serializes the complete inference pipeline with a lock; waiting requests reuse that pair. This also prevents overlapping use of LDIF's shared temporary paths within the single server process. Serialized-serving results are stored separately from the original parallel-inference attempt. No model redesign was added. These finite tests do not establish sustained user capacity.

For gzip measurements, substitute an actual completed smoke directory:

```bash
bash run_4070.sh run_revision_tests.py --suite payload --payload-dir revision_results/<run>/smoke-Total3D --repeats 10
```

Ground-truth test data is not included. These inputs cannot establish 3D accuracy, Quest domain performance, or client energy consumption. Numerical findings remain subject to author review before insertion in the manuscript or rebuttal.
