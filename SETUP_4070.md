# RTX 4070 Laptop: WSL Ubuntu 24.04

This setup is prepared for a fresh `hsu` environment. It has not yet passed inference on the laptop. Do not use the old RTX 5080 upgrade/restore scripts.

## 1. Environment

Clone this repository inside the WSL Linux filesystem, not `/mnt/c`. Install Conda in WSL first. The Windows NVIDIA driver supplies WSL GPU access: do not install an Ubuntu NVIDIA display driver.

```bash
git clone https://github.com/GijsFiten/Edge-XReal.git
cd Edge-XReal
conda env create -f environment-4070.yaml
conda activate hsu
```

`environment-4070.yaml` retains Python 3.8 and the original application's library pins. The approved differences are PyTorch 2.0.1, torchvision 0.15.2, matching torchaudio 2.0.2, CUDA 11.8 runtime/toolkit, and a Conda GCC/G++ 11 compiler. The original `environment.yaml` is unchanged. CUDA, `nvcc` and the compiler are installed inside Conda; no Ubuntu CUDA toolkit installation is required. Creating this environment is a new install, not an instruction to overwrite an existing `hsu`.

The environment is derived from the original complete export, including its optional development/training dependencies. Package solving and installation still need to succeed on the laptop. If installation fails, retain the error and stop; do not loosen versions or use a blanket upgrade. No model retraining is part of this setup.

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

For burst and interval experiments, see `REVISION_TESTS.md`. LDIF currently reuses temporary paths, so concurrent responses can be contaminated even if they pass JSON validation. Those tests characterize the existing implementation; they cannot support a concurrency-capacity claim without additional correctness evidence. No queue or model redesign was added.

For gzip measurements, substitute an actual completed smoke directory:

```bash
bash run_4070.sh run_revision_tests.py --suite payload --payload-dir revision_results/<run>/smoke-Total3D --repeats 10
```

Ground-truth test data is not included. These inputs cannot establish 3D accuracy, Quest domain performance, or client energy consumption. Numerical findings remain subject to author review before insertion in the manuscript or rebuttal.
