from flask import Flask, request, jsonify, Response # type: ignore
from PIL import Image
import argparse
import io
import numpy as np # type: ignore
from time import time
import json
import gzip
import sys
import os
import threading

# Get repo root for later use
_repo_root = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser('Distri3D')
parser.add_argument('--mode', type=str, default='demo', help='train, test, demo_with_time, demo or qtrain, qtest')
parser.add_argument('--demo_path', type=str, default='demo/inputs/1', help='Please specify the demo path.')
parser.add_argument('--save_results', type=str, default='demo/data_time/')
parser.add_argument('--name', type=str, default=None, help='wandb exp name.')
parser.add_argument('--avg_amount', type=int, default=None, help='The amount of samples to run the timing on')
parser.add_argument('--sweep', action='store_true')
parser.add_argument('--hsu_model', type=str, default='3DSceneFormer', help='Total3D, Implicit3D or 3DSceneFormer')
parser.add_argument('--bb_model', type=str, default='Resnet50', help='2D detection model to use: Resnet50, Mobilenetv2, DETR')

#Depending on the --hsu_model argument, the config file and HSU is loaded
_known_args, _ = parser.parse_known_args()
if _known_args.hsu_model == '3DSceneFormer':
    # ThreeDSceneFormer uses relative imports extensively, so import as subpackage
    from ThreeDSceneFormer.configs.config_utils import CONFIG
    from ThreeDSceneFormer.process_API import run, initiate
    # Provide a sensible default config for 3DSceneFormer
    default_config_path = os.path.join(_repo_root, 'ThreeDSceneFormer/configs/transformer_enc_dec_total3d.yaml')

elif _known_args.hsu_model == 'Implicit3D':
    # Add only Implicit3D to sys.path for its internal absolute imports
    _model_dir = os.path.join(_repo_root, 'Implicit3DUnderstanding')
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)
    os.chdir(_model_dir)
    from configs.config_utils import CONFIG # type: ignore
    from process_API import run, initiate # type: ignore
    # Fallback default config for Implicit3D
    default_config_path = os.path.join(_repo_root, 'Implicit3DUnderstanding/configs/total3d_ldif_gcnn.yaml')

elif _known_args.hsu_model == 'Total3D':
    # Add only Total3D to sys.path for its internal absolute imports
    _model_dir = os.path.join(_repo_root, 'Total3DUnderstanding')
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)
    os.chdir(_model_dir)
    from configs.config_utils import CONFIG # type: ignore
    from process_API import run, initiate # type: ignore
    # Fallback default config for Total3D
    default_config_path = os.path.join(_repo_root, 'Total3DUnderstanding/configs/total3d.yaml')

else:
    # Default to 3DSceneFormer config if an unknown option is provided
    from ThreeDSceneFormer.configs.config_utils import CONFIG
    from ThreeDSceneFormer.process_API import run, initiate
    default_config_path = os.path.join(_repo_root, 'ThreeDSceneFormer/configs/transformer_enc_dec_total3d.yaml')

# Ensure CONFIG(...) can access args.config; use a positional with optional default
parser.add_argument('config', type=str, nargs='?', default=default_config_path,
                    help='Path to YAML config. If omitted, a model-specific default is used.')

# Parse arguments and initialize config based on model type
args = parser.parse_args()

if _known_args.hsu_model == 'Total3D':
    # Total3D: CONFIG takes a file path and needs manual update
    cfg = CONFIG(args.config)
    cfg.update_config(args.__dict__)
    # Import and call initiate_environment
    from net_utils.utils import initiate_environment # type: ignore
    initiate_environment(cfg.config)
else:
    # 3DSceneFormer and Implicit3D: CONFIG takes a parser
    cfg = CONFIG(parser)

cfg.config['mode'] = 'demo'
cfg.config['bb_model'] = _known_args.bb_model
initiate(cfg)

# One detector/HSU pair is initialized above and reused for every request.
# Serialize the complete pipeline, including LDIF's shared temporary files.
_inference_lock = threading.Lock()
inference_policy = 'one_loaded_detector_and_hsu_pair; serialized_inference'

preprocess = []
calculations = []

app = Flask(__name__)


@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    """Health check endpoint to verify server is running and accessible."""
    return '', 200


@app.route('/upload', methods=['POST'])
def upload_photo():
    start = time()
    if 'photo' not in request.files:
        return jsonify({'error': 'No file part'})
    if 'cam' not in request.files:
        return jsonify({'error': 'No file part'})
    
    photo = request.files['photo']
    cam = request.files["cam"]

    buffer_content = cam.read()

    content_str = buffer_content.decode('utf-8')
    data = [[float(num) for num in line.split()] for line in content_str.split('\n') if line.strip()]
    array = np.array(data)

    stream = io.BytesIO(photo.read())
    image = Image.open(stream).convert("RGB")

    preprocess.append(time() - start)
    #print data to asses response time
    #print("preprocess = " + str(preprocess))
   
    with _inference_lock:
        compressed_data = run(image,array)
    if(compressed_data == False):
        return jsonify({'error': 'No 2D detections'})
     
    return compressed_data
if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)


