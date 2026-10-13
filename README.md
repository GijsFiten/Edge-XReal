# XR-Scene-Understanding (XRSU)

Repository accompanying “XR-Scene-Understanding: Distributed Deep Learning Based Holistic Scene Understanding for Extended Reality Applications”.

This code provides a minimal Flask server that exposes HSU pipelines (ThreeDSceneFormer, Total3D, Implicit3D) as an HTTP API for evaluation.

## RTX 4070 Laptop revision tests

For a fresh WSL Ubuntu 24.04 installation with CUDA inside Conda, follow [SETUP_4070.md](SETUP_4070.md). This path includes asset verification, an inference-only LDIF build and the revision measurement runner. Laptop runtime validation is still pending.

## Original reported environment

- OS: Ubuntu 24.04 LTS
- Python: 3.8
- CUDA: 11.8 (nvcc required)
- GPU: NVIDIA GTX 1070 or RTX 4070

## Quickstart
Make sure you have CUDA-toolkit and nvcc installed on your system before proceeding.

```bash
sudo apt update
sudo apt install -y xvfb ninja-build freeglut3-dev libglew-dev meshlab
conda env create -f environment.yaml
conda activate hsu
chmod +x ./build.sh
./build.sh
```
- If build errors mention compute capability, adjust it in CMakeLists.txt under external/ldif folders and rebuild.

3) Model data and checkpoints
- Download XRSU.zip from [Google Drive](https://drive.google.com/file/d/19FCQY0RVHrqbT7aR5_y30POhe3Z-v6iF/view?usp=sharing)
- Extract into the project root so files land in the expected subdirectories.

## Run

The server exposes a minimal API for evaluation.

- HSU models: ThreeDSceneFormer, Total3DUnderstanding, Implicit3DUnderstanding
- 2D detectors: Resnet50, Mobilenetv2, DETR

Start server (0.0.0.0 by default):
```bash
conda activate hsu
python server.py --hsu_model ThreeDSceneFormer --bb_model Resnet50
```

- Otherwise, set FLASK run host in your launcher or modify server.py to use host="0.0.0.0". If no parameters are passed, 3D-Scene-Former and Resnet50 will be used.

## API

```/health```
- Method: GET or HEAD
- Response: 200 with empty body (health check)

```/upload```
- Method: POST (multipart/form-data)
- Fields:
    - photo: RGB image file (e.g., .jpg/.png)
    - cam: text file containing the 3x3 camera intrinsic matrix K (space-separated rows)
- Response: 200 with gzipped result payload on success; 4xx/5xx on errors

Examples:
```bash
# Health
curl -I http://localhost:5000/health

# Upload (Linux/macOS)
curl -X POST http://localhost:5000/upload \
  -F "photo=@sample_images/0001.jpg" \
  -F "cam=@sample_images/0001_K.txt" \
  -o result.gz
```

## Client

A simple Python client is included to batch over sample_images and POST to /upload for performance tests. Any HTTP client that sends the two fields above will work.

## Unity

A Unity project is provided as a VR proof of concept. The program will take a screenshot using the Meta Quest 3 internal camera, communicate with the flask application (can be remotely) and display the returned 3D Meshes in VR. The project can be downloaded on [Google Drive](https://drive.google.com/file/d/1jr4Ith_I_vYVc35jzt5iaqhLaGvhviXB/view?usp=sharing). 

## Troubleshooting

- ldif2mesh not found or exit status 127
  - Ensure it exists and is executable:
    `ThreeDSceneFormer/external/ldif/ldif2mesh/ldif2mesh`
  - Re-run build.sh after adjusting compute capability in external/ldif CMakeLists if needed.

- Missing checkpoints or data
  - Ensure you extracted XRSU.zip at the project root so referenced weight paths in configs exist.

## Cite

Citation details will be added upon publication.
