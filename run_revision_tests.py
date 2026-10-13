#!/usr/bin/env python3
"""Edge-XReal measurement runner. Run from conda hsu; see REVISION_TESTS.md.

Uses the existing server and model inference, without changing network/queue semantics.
Success means a structurally valid response, NOT accurate reconstruction.
"""
import argparse
import concurrent.futures
import contextlib
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

MODELS = {
    'Total3D': ('Total3DUnderstanding', 'total3d.yaml'),
    'Implicit3D': ('Implicit3DUnderstanding', 'total3d_ldif_gcnn.yaml'),
    '3DSceneFormer': ('ThreeDSceneFormer', 'transformer_enc_dec_total3d.yaml'),
}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command_info(command, cwd):
    try:
        p = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, timeout=20)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def doctor(args, out):
    result = {'python': sys.version, 'executable': sys.executable,
              'conda_environment': os.environ.get('CONDA_DEFAULT_ENV'),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'cuda_home': os.environ.get('CUDA_HOME'),
              'nvidia_smi': command_info(['nvidia-smi'], args.repo),
              'nvcc': command_info(['nvcc', '--version'], args.repo)}
    try:
        import torch
        result.update(torch=torch.__version__, torch_cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), torch_arch_list=torch.cuda.get_arch_list())
        if torch.cuda.is_available():
            result['gpu'] = torch.cuda.get_device_name()
            result['capability'] = torch.cuda.get_device_capability()
            tensor = torch.ones((32, 32), device='cuda')
            value = (tensor @ tensor).sum().item()
            torch.cuda.synchronize()
            result['cuda_kernel_probe'] = 'passed' if value == 32768 else 'unexpected result'
        else:
            result['cuda_kernel_probe'] = 'not run: CUDA unavailable to PyTorch'
    except Exception:
        result['cuda_kernel_error'] = traceback.format_exc()
    save_json(out / 'environment_check.json', result)
    print(json.dumps(result, indent=2), flush=True)


def samples(root, limit):
    folders = sorted((p for p in root.iterdir() if p.is_dir()),
                     key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))
    if not folders:
        raise ValueError('No sample folders found: ' + str(root))
    if 0 < limit < len(folders):
        indices = [round(i * (len(folders) - 1) / max(1, limit - 1)) for i in range(limit)]
        folders = [folders[i] for i in indices]
    records = []
    for folder in folders:
        photo, camera = folder / 'img.jpg', folder / 'cam_K.txt'
        if not photo.is_file() or not camera.is_file():
            raise FileNotFoundError('Expected img.jpg and cam_K.txt in ' + str(folder))
        rows = [list(map(float, line.split())) for line in camera.read_text().splitlines() if line.strip()]
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError('Camera matrix must be 3x3: ' + str(camera))
        if not all(math.isfinite(v) for row in rows for v in row) or rows[0][0] <= 0 or rows[1][1] <= 0:
            raise ValueError('Invalid camera matrix: ' + str(camera))
        records.append({'id': folder.name, 'photo': str(photo), 'camera': str(camera),
                        'photo_sha256': digest(photo), 'camera_sha256': digest(camera)})
    return records


def prepare(args, out):
    import yaml
    root = args.repo.resolve()
    package, filename = MODELS[args.model]
    config_path = (args.config or root / package / 'configs' / filename).resolve()
    config = yaml.safe_load(config_path.read_text())
    weights = config.get('weight', [])
    weights = [weights] if isinstance(weights, str) else weights
    if not weights:
        raise ValueError('No checkpoint configured')
    # Resolve paths explicitly; do not silently select a different trained model.
    weight_paths = [(root / package / p).resolve() for p in weights]
    detector = root / '2D_Detectors/Resnet50/fine_tuned_resnet.pth'
    priors = [root / package / 'data/sunrgbd/preprocessed' / name
              for name in ('layout_avg_file.pkl', 'size_avg_category.pkl')]
    missing = [str(p) for p in weight_paths + [detector] + priors if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing inference assets (inference refused): ' + ', '.join(missing))
    config['weight'] = [str(p) for p in weight_paths]
    config['resume'] = False
    config['finetune'] = True
    config['log']['path'] = str(out / 'model_logs')
    config['log']['vis_path'] = 'visualization'
    config['device']['use_gpu'] = True
    effective = out / 'effective_config.yaml'
    effective.write_text(yaml.safe_dump(config), encoding='utf-8')
    limit = args.limit if args.limit is not None else (5 if args.suite == 'smoke' else 20 if args.suite == 'intervals' else 0)
    chosen = samples((args.inputs or root / 'sample_images').resolve(), limit)
    manifest = {'status': 'prepared_not_run', 'model': args.model, 'detector': 'Resnet50',
                'suite': args.suite, 'command': sys.argv, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'allow_no_detections': getattr(args, 'allow_no_detections', False),
                'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                'conda_environment': os.environ.get('CONDA_DEFAULT_ENV'),
                'repo': str(root), 'source_config': str(config_path), 'effective_config': config,
                'source_config_sha256': digest(config_path),
                'checkpoints': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                                for p in weight_paths + [detector]],
                'priors': [{'path': str(p), 'sha256': digest(p)} for p in priors],
                'samples': chosen, 'warmup': 0 if args.suite == 'smoke' else args.warmup,
                'repeats': 1 if args.suite == 'smoke' else args.repeats,
                'git_head': command_info(['git', 'rev-parse', 'HEAD'], root),
                'git_status': command_info(['git', 'status', '--porcelain'], root),
                'nvidia_smi': command_info(['nvidia-smi'], root),
                'gpu_conditions': command_info(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,power.limit,power.draw,temperature.gpu,clocks.sm,clocks.mem', '--format=csv'], root),
                'cpu': command_info(['lscpu'], root),
                'memory': command_info(['free', '-b'], root),
                'packages': command_info([sys.executable, '-m', 'pip', 'freeze'], root)}
    save_json(out / 'manifest.json', manifest)
    (out / 'working_tree.patch').write_text(command_info(['git', 'diff', '--no-ext-diff'], root).get('stdout', ''), encoding='utf-8')
    return chosen, effective


def load_server(args, config):
    """Install an audit around the original checkpoint loader before server initialisation."""
    root = args.repo.resolve()
    package = MODELS[args.model][0]
    os.chdir(root)
    sys.path.insert(0, str(root))
    if args.model == '3DSceneFormer':
        utils = importlib.import_module(package + '.net_utils.utils')
    else:
        sys.path.insert(0, str(root / package))
        os.chdir(root / package)
        utils = importlib.import_module('net_utils.utils')
    loaded_keys = set()
    original_parse = utils.CheckpointIO.parse_state_dict

    def audited_parse(self, checkpoint, *domain):
        weights = checkpoint.get('net', {})
        net = self.module_dict.get('net')
        if net is not None:
            current = net.state_dict()
            loaded_keys.update(k for k, v in weights.items()
                               if k in current and getattr(v, 'shape', None) == current[k].shape)
        return original_parse(self, checkpoint, *domain)

    utils.CheckpointIO.parse_state_dict = audited_parse
    sys.argv = [str(root / 'server.py'), '--hsu_model', args.model, '--bb_model', 'Resnet50', str(config)]
    # Model folders also contain server.py. Load the repository API explicitly
    # after adding a model folder to sys.path for its internal imports.
    spec = importlib.util.spec_from_file_location('server', root / 'server.py')
    server = importlib.util.module_from_spec(spec)
    sys.modules['server'] = server
    spec.loader.exec_module(server)
    api = sys.modules[server.run.__module__]
    required_keys = {k for k in api.net.state_dict() if not k.endswith('.num_batches_tracked')}
    missing = sorted(required_keys - loaded_keys)
    if missing:
        raise RuntimeError('Checkpoint did not load every inference parameter/buffer; first missing keys: ' + ', '.join(missing[:15]))
    import torch
    import torchvision
    if not torch.cuda.is_available() or str(api.device).split(':')[0] != 'cuda':
        raise RuntimeError('GPU inference required; no CUDA model device')
    torch.cuda.synchronize()
    info = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(), 'device_capability': torch.cuda.get_device_capability(),
            'torch_arch_list': torch.cuda.get_arch_list(), 'loaded_state_keys': len(loaded_keys),
            'checkpoint_audit_exception': 'BatchNorm num_batches_tracked counters are not required for eval-mode inference',
            'inference_policy': server.inference_policy,
            'server_pid': os.getpid(),
            'effective_runtime_config': server.cfg.config,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'attention_backends': {'flash': torch.backends.cuda.flash_sdp_enabled(),
                                   'memory_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
                                   'math': torch.backends.cuda.math_sdp_enabled()}}
    info['inference_source_sha256'] = digest(Path(api.__file__))
    info['server_source_sha256'] = digest(root / 'server.py')
    return server, info


def serve(args):
    started = time.perf_counter()
    server, info = load_server(args, args.config)
    from werkzeug.serving import make_server
    httpd = make_server('127.0.0.1', 0, server.app, threaded=True)
    info.update(url='http://127.0.0.1:' + str(httpd.server_port),
                init_seconds=time.perf_counter() - started,
                boundary='threaded Werkzeug HTTP on WSL loopback; excludes XR capture/render and physical network')
    save_json(args.out / 'ready.json', info)
    httpd.serve_forever()


@contextlib.contextmanager
def managed_server(args, out, config):
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--suite', '_serve',
           '--repo', str(args.repo.resolve()), '--model', args.model, '--config', str(config), '--out', str(out)]
    with (out / 'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, cwd=str(args.repo.resolve()), stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while not (out / 'ready.json').is_file():
                if process.poll() is not None:
                    raise RuntimeError('Server initialisation failed; see ' + str(out / 'server.log'))
                if time.monotonic() > deadline:
                    raise TimeoutError('Server startup timed out; see server.log')
                time.sleep(0.2)
            yield json.loads((out / 'ready.json').read_text())['url']
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def validate_payload(payload):
    started = time.perf_counter()
    raw = gzip.decompress(payload)
    decoded = time.perf_counter()
    scene = json.loads(raw)
    parsed = time.perf_counter()
    for key in ('layout', 'cam_R', 'class_id', 'bdb', 'objects'):
        if key not in scene:
            raise ValueError('Missing scene field: ' + key)
    n = len(scene['objects'])
    if n == 0 or len(scene['bdb']) != n or len(scene['class_id']) != n:
        raise ValueError('Empty or inconsistent object counts')

    def matrix(value, rows, columns):
        if not isinstance(value, list) or (rows is not None and len(value) != rows) or not value:
            raise ValueError('Invalid matrix shape')
        if any(not isinstance(row, list) or len(row) != columns for row in value):
            raise ValueError('Invalid matrix shape')
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for row in value for x in row):
            raise ValueError('Non-finite/non-numeric geometry')

    matrix(scene['layout'], 8, 3)
    matrix(scene['cam_R'], 3, 3)
    for box, mesh in zip(scene['bdb'], scene['objects']):
        matrix(box['basis'], 3, 3)
        matrix([box['coeffs'], box['centroid']], 2, 3)
        matrix(mesh['v'], None, 3)
        matrix(mesh['f'], None, 3)
        # Existing API uses OBJ faces (one-based). Do not silently reindex them.
        if any(int(v) != v or v < 1 or v > len(mesh['v']) for face in mesh['f'] for v in face):
            raise ValueError('Invalid OBJ triangle vertex index')
    return {'gzip_bytes': len(payload), 'json_bytes': len(raw), 'object_count': n,
            'vertices': sum(len(m['v']) for m in scene['objects']),
            'triangles': sum(len(m['f']) for m in scene['objects']),
            'decompress_ms': 1000 * (decoded - started), 'json_parse_ms': 1000 * (parsed - decoded),
            'scene_json_sha256': hashlib.sha256(raw).hexdigest()}


def request_sample(url, sample, timeout, barrier=None, scheduled_at=None):
    # Input files and multipart construction are outside the HTTP measurement.
    boundary = 'edge_xreal_revision_boundary'
    parts = []
    for name, key, mime in [('photo', 'photo', 'image/jpeg'), ('cam', 'camera', 'text/plain')]:
        p = Path(sample[key])
        parts.extend([('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name +
                       '"; filename="' + p.name + '"\r\nContent-Type: ' + mime + '\r\n\r\n').encode(),
                      p.read_bytes(), b'\r\n'])
    parts.append(('--' + boundary + '--\r\n').encode())
    body = b''.join(parts)
    request = urllib.request.Request(url + '/upload', body,
                                    {'Content-Type': 'multipart/form-data; boundary=' + boundary})
    record = {'sample_id': sample['id'], 'input_multipart_bytes': len(body), 'status': 'error'}
    if barrier is not None:
        barrier.wait(timeout=timeout)
    started = time.perf_counter()
    if scheduled_at is not None:
        record['dispatch_lag_ms'] = 1000 * (started - scheduled_at)
    payload = None
    try:
        # No retries: every failed/timed-out request remains in the denominator.
        with urllib.request.urlopen(request, timeout=timeout) as response:
            record['http_status'] = response.status
            payload = response.read()
        record['response_ms'] = 1000 * (time.perf_counter() - started)
        if not payload.startswith(b'\x1f\x8b'):
            if record['http_status'] == 200:
                try:
                    if json.loads(payload) == {'error': 'No 2D detections'}:
                        record['outcome'] = 'no_detections'
                except (ValueError, UnicodeDecodeError):
                    pass
            raise ValueError('Non-gzip API response: ' + payload[:300].decode('utf-8', errors='replace'))
        record.update(validate_payload(payload))
        record['status'] = 'ok'
    except Exception as exc:
        record.setdefault('response_ms', 1000 * (time.perf_counter() - started))
        if isinstance(exc, urllib.error.HTTPError):
            record['http_status'] = exc.code
        record['error'] = type(exc).__name__ + ': ' + str(exc)
        timeout_types = (TimeoutError, socket.timeout)
        record['timed_out'] = isinstance(exc, timeout_types) or isinstance(getattr(exc, 'reason', None), timeout_types)
    return record, payload


def payload_tests(args, out):
    if args.payload_dir is None:
        raise ValueError('--payload-dir must point to a completed smoke run containing scene-*.json.gz')
    paths = sorted(args.payload_dir.glob('scene-*.json.gz'))
    if not paths:
        raise ValueError('No saved scene payloads found')
    with (out / 'compression.jsonl').open('w') as stream:
        for path in paths:
            payload = path.read_bytes()
            validate_payload(payload)
            raw = gzip.decompress(payload)
            for level in (1, 6, 9):
                for repeat in range(args.repeats):
                    start = time.perf_counter()
                    packed = gzip.compress(raw, compresslevel=level)
                    compression_ms = 1000 * (time.perf_counter() - start)
                    row = {'source': str(path.resolve()), 'source_sha256': digest(path), 'repeat': repeat,
                           'gzip_level': level, 'compression_ms': compression_ms, **validate_payload(packed)}
                    stream.write(json.dumps(row) + '\n')
    save_json(out / 'manifest.json', {'suite': 'payload', 'python': sys.version, 'platform': platform.platform(),
                                    'boundary': 'CPU recompression/decompression of the same JSON bytes; no inference or network',
                                    'repeats': args.repeats, 'source_count': len(paths)})


def summarise(records):
    selected = [r for r in records if not r.get('warmup')]
    ok = [r['response_ms'] for r in selected if r['status'] == 'ok']
    values = sorted(ok)
    no_detections = sum(r.get('outcome') == 'no_detections' for r in selected)
    return {'attempted': len(selected), 'successful_valid_responses': len(ok),
            'failed': len(selected) - len(ok),
            'no_detections': no_detections,
            'other_failures': len(selected) - len(ok) - no_detections,
            'success_latency_mean_ms': statistics.mean(ok) if ok else None,
            'success_latency_median_ms': statistics.median(ok) if ok else None,
            'success_latency_p95_nearest_rank_ms': values[math.ceil(.95 * len(values)) - 1] if values else None,
            'warning': 'Latency is conditional on valid responses. No reconstruction accuracy, headset latency, or user capacity is inferred.'}


def accepted_outcome(row, args):
    return row['status'] == 'ok' or (
        getattr(args, 'allow_no_detections', False) and row.get('outcome') == 'no_detections')


def local_sample(server, sample, worker_start=None, input_bytes=None):
    import numpy as np
    import torch
    from PIL import Image
    photo, camera = input_bytes or (Path(sample['photo']).read_bytes(), Path(sample['camera']).read_bytes())
    row = {'sample_id': sample['id'], 'status': 'error'}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(photo)).convert('RGB')
        matrix = np.array([[float(v) for v in line.split()] for line in camera.decode('utf-8').splitlines() if line.strip()])
        payload = server.run(image, matrix)
        torch.cuda.synchronize()
        completed = time.perf_counter()
        row['response_ms'] = 1000 * (completed - start)
        if worker_start is not None:
            row['worker_start_to_output_ms'] = 1000 * (completed - worker_start)
        if payload is False:
            row['outcome'] = 'no_detections'
            raise ValueError('No 2D detections')
        row.update(validate_payload(payload))
        row['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        row['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        row['status'] = 'ok'
    except Exception as exc:
        row.setdefault('response_ms', 1000 * (time.perf_counter() - start))
        row['error'] = type(exc).__name__ + ': ' + str(exc)
    return row


def local_tests(args, out, chosen, config):
    start = time.perf_counter()
    server, info = load_server(args, config)
    info['init_seconds'] = time.perf_counter() - start
    info['boundary'] = 'local RGB/camera decoding through synchronised gzip-JSON output; input file reads and client validation excluded'
    save_json(out / 'runtime.json', info)
    api = sys.modules[server.run.__module__]
    if args.suite == 'profile':
        import torch
        def synchronised_clock():
            torch.cuda.synchronize()
            return time.perf_counter()
        # Diagnostic pass only: existing stage boundaries now wait for GPU completion.
        # Do not use these instrumented totals as the main latency comparison.
        api.time = synchronised_clock
    rows = []
    with (out / 'requests.jsonl').open('w') as stream:
        schedule = [(True, 0, chosen[i % len(chosen)]) for i in range(args.warmup)]
        schedule += [(False, r, sample) for r in range(args.repeats) for sample in chosen]
        for warmup, repeat, sample in schedule:
            before = [len(getattr(api, name)) for name in ('twodet', 'implicit', 'zip_time')]
            row = local_sample(server, sample)
            if args.suite == 'profile' and row['status'] == 'ok':
                stages = ('detection_and_preparation_ms', 'hsu_forward_and_mesh_ms', 'output_postprocess_json_gzip_ms')
                for name, stage, count in zip(('twodet', 'implicit', 'zip_time'), stages, before):
                    values = getattr(api, name)
                    if len(values) != count + 1:
                        raise RuntimeError('Unexpected stage counters: ' + name)
                    row[stage] = 1000 * values[-1]
            row.update(warmup=warmup, repeat=repeat, phase=args.suite, concurrency=1)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            print('local ' + sample['id'] + ': ' + row['status'], flush=True)
            if warmup and not accepted_outcome(row, args):
                raise RuntimeError('Local warmup failed')
    save_json(out / 'summary.json', summarise(rows))
    if any(not accepted_outcome(r, args) for r in rows):
        raise RuntimeError('Local inference failures; see requests.jsonl')
    return rows


def cold_worker(args):
    sample = {'id': args.worker_sample.name, 'photo': str(args.worker_sample / 'img.jpg'),
              'camera': str(args.worker_sample / 'cam_K.txt')}
    input_bytes = (Path(sample['photo']).read_bytes(), Path(sample['camera']).read_bytes())
    started = time.perf_counter()
    server, info = load_server(args, args.config)
    init_seconds = time.perf_counter() - started
    row = local_sample(server, sample, worker_start=started, input_bytes=input_bytes)
    row['init_seconds'] = init_seconds
    save_json(args.out / 'result.json', row)
    save_json(args.out / 'runtime.json', info)


def cold_tests(args, out, chosen, config):
    rows = []
    with (out / 'requests.jsonl').open('w') as stream:
        for repeat in range(args.repeats):
            for index, sample in enumerate(chosen):
                child = out / ('cold-%03d-%03d' % (repeat, index))
                child.mkdir()
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--suite', '_cold_worker',
                       '--repo', str(args.repo.resolve()), '--model', args.model, '--config', str(config),
                       '--out', str(child), '--worker-sample', str(Path(sample['photo']).parent)]
                started = time.perf_counter()
                with (child / 'worker.log').open('w') as log:
                    try:
                        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                 timeout=args.startup_timeout + args.timeout)
                        wall_ms = 1000 * (time.perf_counter() - started)
                        if process.returncode != 0 or not (child / 'result.json').is_file():
                            raise RuntimeError('Cold worker failed; see ' + str(child / 'worker.log'))
                        row = json.loads((child / 'result.json').read_text())
                        row['pipeline_ms'] = row.pop('response_ms')
                        row['process_wall_ms'] = wall_ms
                        # Primary cold measure includes initialisation in the worker, not interpreter startup/shutdown.
                        row['response_ms'] = row.get('worker_start_to_output_ms', wall_ms)
                    except Exception as exc:
                        row = {'sample_id': sample['id'], 'status': 'error',
                               'response_ms': 1000 * (time.perf_counter() - started),
                               'error': type(exc).__name__ + ': ' + str(exc)}
                row.update(warmup=False, repeat=repeat, phase='cold', concurrency=1)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print('cold ' + sample['id'] + ': ' + row['status'], flush=True)
                if not accepted_outcome(row, args):
                    raise RuntimeError('Cold worker failed; resolve the failure before repeated initialisation')
    save_json(out / 'summary.json', summarise(rows))
    return rows


def interval_tests(args, chosen, url, out, record):
    with (out / 'interval_trials.jsonl').open('w') as trials:
        for interval in args.intervals:
            for repeat in range(args.repeats):
                started = time.perf_counter()
                results = []
                # One finite trial, no hidden client request queue. At most len(chosen) requests.
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(chosen)) as pool:
                    futures = []
                    for index, sample in enumerate(chosen):
                        deadline = started + index * interval
                        time.sleep(max(0, deadline - time.perf_counter()))
                        futures.append(pool.submit(request_sample, url, sample, args.timeout,
                                                   scheduled_at=deadline))
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        result[0]['interval_seconds'] = interval
                        result[0]['phase'] = 'intervals'
                        record(result, 'intervals', repeat, None)
                        results.append(result[0])
                duration = time.perf_counter() - started
                trials.write(json.dumps({'interval_seconds': interval, 'repeat': repeat,
                                         'trial_wall_seconds': duration,
                                         'valid_responses_per_second': sum(r['status'] == 'ok' for r in results) / duration,
                                         **summarise(results)}) + '\n')
                trials.flush()
                if any(r.get('timed_out') for r in results):
                    raise RuntimeError('Interval trial timed out; stopping before another trial overlaps unfinished work')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['doctor', 'preflight', 'smoke', 'local', 'cold', 'http', 'profile', 'load', 'intervals', 'payload', '_serve', '_cold_worker'], default='smoke')
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--model', choices=MODELS, default='Total3D')
    p.add_argument('--config', type=Path)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--payload-dir', type=Path)
    p.add_argument('--worker-sample', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--out', type=Path)
    p.add_argument('--limit', type=int, help='0 = all input folders; smoke default 5, others all')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--allow-no-detections', action='store_true',
                   help='Continue on the exact no-detection outcome; still count it separately from valid scenes')
    p.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--intervals', type=float, nargs='+', default=[0.5, 1, 2, 3])
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--startup-timeout', type=float, default=600)
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0 or (args.limit is not None and args.limit < 0):
        p.error('Require repeats >= 1, warmup >= 0 and limit >= 0')
    if min(args.concurrency) < 1 or args.timeout <= 0 or args.startup_timeout <= 0:
        p.error('Concurrency and timeouts must be positive')
    if min(args.intervals) <= 0:
        p.error('Intervals must be positive; use load for simultaneous bursts')
    if args.suite == '_serve':
        serve(args)
        return
    if args.suite == '_cold_worker':
        cold_worker(args)
        return
    out = (args.out or args.repo / 'revision_results' / (time.strftime('%Y%m%d-%H%M%S') + '-' + args.model + '-' + args.suite)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print('Results: ' + str(out), flush=True)
    try:
        if args.suite == 'doctor':
            doctor(args, out)
            return
        if args.suite == 'payload':
            payload_tests(args, out)
            print('Payload CPU tests completed; see compression.jsonl', flush=True)
            return
        chosen, config = prepare(args, out)
        if args.suite == 'preflight':
            print('Files checked; no inference performed.', flush=True)
            return
        if os.environ.get('CONDA_DEFAULT_ENV', '').split('/')[-1] != 'hsu':
            raise RuntimeError('Activate conda hsu before running inference')
        if args.suite in ('local', 'profile', 'cold'):
            rows = (cold_tests if args.suite == 'cold' else local_tests)(args, out, chosen, config)
            save_json(out / 'run_status.json', {'completed': True,
                       'failures': sum(r['status'] != 'ok' for r in rows if not r.get('warmup')),
                       'no_detections': sum(r.get('outcome') == 'no_detections' for r in rows if not r.get('warmup')),
                       'allow_no_detections': args.allow_no_detections})
            return
        records = []
        with managed_server(args, out, config) as url, (out / 'requests.jsonl').open('w') as stream:
            def record(result, phase, repeat, concurrency, requested_concurrency=None):
                row, payload = result
                row.update(phase=phase, warmup=(phase == 'warmup'), repeat=repeat, concurrency=concurrency,
                           requested_concurrency=requested_concurrency or concurrency)
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                if payload is not None and phase == 'smoke' and row['status'] == 'ok':
                    (out / ('scene-' + row['sample_id'] + '.json.gz')).write_bytes(payload)
                print(phase + ' ' + row['sample_id'] + ': ' + row['status'], flush=True)
            warmup = 0 if args.suite == 'smoke' else args.warmup
            for i in range(warmup):
                result = request_sample(url, chosen[i % len(chosen)], args.timeout)
                record(result, 'warmup', i, 1)
                if not accepted_outcome(result[0], args):
                    raise RuntimeError('Warmup failed; resolve smoke-test failure before benchmarking')
            repeats = 1 if args.suite == 'smoke' else args.repeats
            if args.suite == 'intervals':
                interval_tests(args, chosen, url, out, record)
            for concurrency in (args.concurrency if args.suite == 'load' else [] if args.suite == 'intervals' else [1]):
                for repeat in range(repeats):
                    # A bounded burst at a time: this tests burst response, not an arrival-rate/user-capacity model.
                    for begin in range(0, len(chosen), concurrency):
                        batch = chosen[begin:begin + concurrency]
                        barrier = threading.Barrier(len(batch))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                            futures = [pool.submit(request_sample, url, sample, args.timeout, barrier)
                                       for sample in batch]
                            timed_out = False
                            for future in futures:
                                result = future.result()
                                record(result, args.suite, repeat, len(batch), concurrency)
                                timed_out |= result[0].get('timed_out', False)
                            if timed_out:
                                raise RuntimeError('Request timeout: stopping so unfinished GPU work cannot contaminate the next batch')
            if args.suite == 'intervals':
                summary = {str(i): summarise([r for r in records if r.get('interval_seconds') == i]) for i in args.intervals}
            else:
                summary = {str(c): summarise([r for r in records if r['concurrency'] == c])
                           for c in sorted(set(r['concurrency'] for r in records))}
            save_json(out / 'summary.json', summary)
        failures = sum(r['status'] != 'ok' for r in records)
        save_json(out / 'run_status.json', {'completed': True, 'failures': failures,
                  'no_detections': sum(r.get('outcome') == 'no_detections' for r in records),
                  'allow_no_detections': args.allow_no_detections})
        print(json.dumps(summary, indent=2), flush=True)
        if any(not accepted_outcome(r, args) for r in records):
            raise RuntimeError(str(failures) + ' invalid/failed requests; inspect requests.jsonl')
    except Exception as exc:
        save_json(out / 'failure.json', {'error': type(exc).__name__ + ': ' + str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
