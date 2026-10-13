"""CPU checks of the measurement harness, not neural-network validation."""
import ast
import copy
import gzip
import importlib.util
import json
import socket
from pathlib import Path
import tempfile
import threading
import types
import sys
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / 'revision_results' / 'unit-test-temp'
SCRATCH.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('runner', REPO / 'run_revision_tests.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def scene():
    return {'layout': [[0., 0., 0.]] * 8, 'cam_R': [[1., 0., 0.]] * 3,
            'class_id': [1], 'bdb': [{'basis': [[1., 0., 0.]] * 3,
                                    'coeffs': [1., 1., 1.], 'centroid': [0., 0., 0.]}],
            'objects': [{'nameObject': '0_1.obj', 'v': [[0., 0., 0.]] * 3, 'f': [[1, 2, 3]]}]}


def pack(value):
    return gzip.compress(json.dumps(value).encode())


class Checks(unittest.TestCase):
    def test_valid_payload_byte_counts(self):
        payload = pack(scene())
        result = runner.validate_payload(payload)
        self.assertEqual(result['gzip_bytes'], len(payload))
        self.assertEqual(result['json_bytes'], len(gzip.decompress(payload)))
        self.assertEqual(result['object_count'], 1)

    def test_nonfinite_geometry_rejected(self):
        value = scene()
        value['objects'][0]['v'][0][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            runner.validate_payload(pack(value))

    def test_face_range_rejected(self):
        value = scene()
        value['objects'][0]['f'][0][2] = 4
        with self.assertRaisesRegex(ValueError, 'vertex index'):
            runner.validate_payload(pack(value))

    def test_inconsistent_counts_rejected(self):
        value = scene()
        value['bdb'] = []
        with self.assertRaisesRegex(ValueError, 'object counts'):
            runner.validate_payload(pack(value))

    def test_failures_count_and_warmup_excluded(self):
        rows = [{'status': 'ok', 'response_ms': 999, 'warmup': True},
                {'status': 'ok', 'response_ms': 10}, {'status': 'error', 'response_ms': 1000}]
        result = runner.summarise(rows)
        self.assertEqual(result['attempted'], 2)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['success_latency_mean_ms'], 10)

    def test_all_failures_no_latency(self):
        result = runner.summarise([{'status': 'error', 'response_ms': 100}])
        self.assertIsNone(result['success_latency_mean_ms'])
        self.assertEqual(result['failed'], 1)

    def test_numeric_sample_selection_and_missing_files(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH.resolve())) as temporary:
            root = Path(temporary)
            assert root.resolve().parent == SCRATCH.resolve()
            for i in range(1, 101):
                p = root / str(i)
                p.mkdir()
                (p / 'img.jpg').write_bytes(str(i).encode())
                (p / 'cam_K.txt').write_text('2 0 1\n0 2 1\n0 0 1\n')
            self.assertEqual([s['id'] for s in runner.samples(root, 5)], ['1', '26', '51', '75', '100'])
            (root / '1' / 'cam_K.txt').unlink()
            with self.assertRaises(FileNotFoundError):
                runner.samples(root, 5)

    def test_http_200_error_is_failure_and_valid_gzip_is_success(self):
        state = {'payload': b'{"error":"No 2D detections"}'}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(state['payload'])
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=str(SCRATCH.resolve())) as temporary:
                root = Path(temporary)
                assert root.resolve().parent == SCRATCH.resolve()
                (root / 'img.jpg').write_bytes(b'test image')
                (root / 'cam_K.txt').write_text('2 0 1\n0 2 1\n0 0 1\n')
                sample = {'id': 'test', 'photo': str(root / 'img.jpg'), 'camera': str(root / 'cam_K.txt')}
                url = 'http://127.0.0.1:' + str(server.server_port)
                record, _ = runner.request_sample(url, sample, 2)
                if 'WinError 10013' in record.get('error', ''):
                    self.skipTest('Windows sandbox blocks loopback connections; no HTTP transport validation')
                self.assertEqual(record['status'], 'error')
                self.assertEqual(record.get('outcome'), 'no_detections')
                self.assertEqual(record.get('http_status'), 200, record)
                state['payload'] = pack(scene())
                record, payload = runner.request_sample(url, sample, 2)
                self.assertEqual(record['status'], 'ok')
                self.assertEqual(record['gzip_bytes'], len(payload))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_response_handling_without_transport(self):
        class Response:
            status = 200
            def __init__(self, body):
                self.body = body
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return self.body
        with tempfile.TemporaryDirectory(dir=str(SCRATCH.resolve())) as temporary:
            root = Path(temporary)
            assert root.resolve().parent == SCRATCH.resolve()
            (root / 'image').write_bytes(b'fixture')
            sample = {'id': 'fixture', 'photo': str(root / 'image'), 'camera': str(root / 'image')}
            with patch.object(runner.urllib.request, 'urlopen', return_value=Response(b'{"error":"No detections"}')):
                row, _ = runner.request_sample('http://localhost', sample, 1)
                self.assertEqual(row['http_status'], 200)
                self.assertEqual(row['status'], 'error')
                self.assertNotIn('outcome', row)
            with patch.object(runner.urllib.request, 'urlopen', return_value=Response(pack(scene()))):
                row, _ = runner.request_sample('http://localhost', sample, 1)
                self.assertEqual(row['status'], 'ok')

    def test_expected_no_detection_is_not_a_valid_scene_or_an_arbitrary_error(self):
        expected = {'status': 'error', 'outcome': 'no_detections'}
        strict = types.SimpleNamespace(allow_no_detections=False)
        allowed = types.SimpleNamespace(allow_no_detections=True)
        self.assertFalse(runner.accepted_outcome(expected, strict))
        self.assertTrue(runner.accepted_outcome(expected, allowed))
        self.assertFalse(runner.accepted_outcome({'status': 'error', 'error': 'CUDA failure'}, allowed))
        summary = runner.summarise([expected, {'status': 'ok', 'response_ms': 10},
                                    {'status': 'error', 'error': 'CUDA failure'}])
        self.assertEqual(summary['attempted'], 3)
        self.assertEqual(summary['successful_valid_responses'], 1)
        self.assertEqual(summary['failed'], 2)
        self.assertEqual(summary['no_detections'], 1)
        self.assertEqual(summary['other_failures'], 1)
        self.assertEqual(summary['success_latency_mean_ms'], 10)

    def test_python38_socket_timeouts_stop_http_measurements(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH.resolve())) as temporary:
            image = Path(temporary) / 'image'
            image.write_bytes(b'fixture')
            sample = {'id': 'fixture', 'photo': str(image), 'camera': str(image)}
            for error in (socket.timeout('timed out'), runner.urllib.error.URLError(socket.timeout('timed out'))):
                with self.subTest(error=type(error).__name__), patch.object(runner.urllib.request, 'urlopen', side_effect=error):
                    row, _ = runner.request_sample('http://localhost', sample, 1)
                    self.assertEqual(row['status'], 'error')
                    self.assertTrue(row['timed_out'])

    def test_cold_dataset_continues_only_on_allowed_no_detection(self):
        for allow in (False, True):
            with self.subTest(allow=allow), tempfile.TemporaryDirectory(dir=str(SCRATCH.resolve())) as temporary:
                out = Path(temporary)
                args = types.SimpleNamespace(repeats=1, model='Total3D', repo=REPO,
                                             startup_timeout=1, timeout=1, allow_no_detections=allow)
                chosen = [{'id': name, 'photo': str(out / name / 'img.jpg')} for name in ('100', '1')]
                calls = []
                def worker(cmd, **kwargs):
                    sample_id = Path(cmd[cmd.index('--worker-sample') + 1]).name
                    child = Path(cmd[cmd.index('--out') + 1])
                    row = {'sample_id': sample_id, 'status': 'ok', 'response_ms': 10,
                           'worker_start_to_output_ms': 30, 'init_seconds': .02}
                    if sample_id == '100':
                        row.update(status='error', outcome='no_detections', error='ValueError: No 2D detections')
                    (child / 'result.json').write_text(json.dumps(row))
                    calls.append(sample_id)
                    return types.SimpleNamespace(returncode=0)
                with patch.object(runner.subprocess, 'run', side_effect=worker):
                    if allow:
                        rows = runner.cold_tests(args, out, chosen, out / 'config.yaml')
                        self.assertEqual(calls, ['100', '1'])
                        self.assertEqual(len(rows), 2)
                        self.assertEqual(rows[0]['response_ms'], 30)
                        self.assertEqual(json.loads((out / 'summary.json').read_text())['no_detections'], 1)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'Cold worker failed'):
                            runner.cold_tests(args, out, chosen, out / 'config.yaml')
                        self.assertEqual(calls, ['100'])

    def test_checkpoint_paths_cwd_independent_and_missing_is_fatal(self):
        repo = REPO
        for package in runner.MODELS.values():
            source = repo / package[0] / 'net_utils/utils.py'
            tree = ast.parse(source.read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CheckpointIO')
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'finetune')
            namespace = {'os': __import__('os'), '__file__': str(source)}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
            fake = type('Fake', (), {})()
            fake.cfg = type('Config', (), {'config': {'mode': 'demo', 'weight': ['nonexistent-checkpoint.pth']},
                                         'log_string': lambda self, message: None})()
            with self.assertRaises(FileNotFoundError):
                namespace['finetune'](fake)
            if package[0] == 'Total3DUnderstanding':
                fake.cfg.config['weight'] = ['out/pretrained_models/pretrained_model.pth']
                loaded = []
                fake.load = lambda path, domain: loaded.append((Path(path), domain))
                with patch('os.path.isfile', return_value=True):
                    namespace['finetune'](fake)
                self.assertEqual(loaded[0][0], source.parent.parent / fake.cfg.config['weight'][0])

    def test_chamfer_extension_is_lazy_and_uses_absolute_paths(self):
        repo = REPO
        source = repo / 'Total3DUnderstanding/external/pyTorchChamferDistance/chamfer_distance/chamfer_distance.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_get_extension')
        with tempfile.TemporaryDirectory(dir=str(SCRATCH.resolve())) as temporary:
            root = Path(temporary)
            assert root.resolve().parent == SCRATCH.resolve()
            module_dir = root / 'chamfer_distance'
            module_dir.mkdir()
            namespace = {'Path': Path, '__file__': str(module_dir / 'chamfer_distance.py'), '_cd': None}
            fake = types.ModuleType('torch.utils.cpp_extension')
            calls = []
            fake.load = lambda **kwargs: calls.append(kwargs) or 'loaded extension'
            with patch.dict(sys.modules, {'torch.utils.cpp_extension': fake}):
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
                self.assertEqual(calls, [])
                self.assertEqual(namespace['_get_extension'](), 'loaded extension')
                self.assertEqual(namespace['_get_extension'](), 'loaded extension')
                self.assertEqual(len(calls), 1)
                self.assertTrue(Path(calls[0]['build_directory']).is_dir())
                self.assertTrue(all(Path(p).is_absolute() for p in calls[0]['sources']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
