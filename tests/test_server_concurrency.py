"""Verify HTTP requests serialize through a single initialized model pair."""
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch
from PIL import Image


class ServerConcurrency(unittest.TestCase):
    def test_two_requests_share_one_pair_and_never_overlap_inference(self):
        from werkzeug.serving import make_server
        import urllib.request
        repo = Path(__file__).resolve().parents[1]
        state = {'initializations': 0, 'active': 0, 'peak': 0, 'calls': [], 'decoded': 0}
        guard = threading.Lock()
        entered, both_decoded, release = threading.Event(), threading.Event(), threading.Event()

        class Config:
            def __init__(self, parser):
                self.config = {}

        def initiate(cfg):
            state['initializations'] += 1
            state['pair'] = (object(), object())

        def run(image, camera):
            with guard:
                state['active'] += 1
                state['peak'] = max(state['peak'], state['active'])
                state['calls'].append(tuple(id(model) for model in state['pair']))
            entered.set()
            try:
                if not release.wait(5):
                    raise RuntimeError('Test did not release inference')
                return gzip.compress(b'{"fixture":true}')
            finally:
                with guard:
                    state['active'] -= 1

        package = types.ModuleType('ThreeDSceneFormer'); package.__path__ = []
        configs = types.ModuleType('ThreeDSceneFormer.configs'); configs.__path__ = []
        config_module = types.ModuleType('ThreeDSceneFormer.configs.config_utils'); config_module.CONFIG = Config
        api = types.ModuleType('ThreeDSceneFormer.process_API'); api.initiate = initiate; api.run = run
        modules = {m.__name__: m for m in (package, configs, config_module, api)}
        spec = importlib.util.spec_from_file_location('serialized_server_fixture', repo / 'server.py')
        server = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.object(sys, 'argv', [str(repo / 'server.py')]):
            spec.loader.exec_module(server)
        self.assertEqual(state['initializations'], 1)
        real_open = server.Image.open
        def open_image(*args, **kwargs):
            image = real_open(*args, **kwargs)
            with guard:
                state['decoded'] += 1
                if state['decoded'] == 2:
                    both_decoded.set()
            return image
        png = io.BytesIO(); Image.new('RGB', (2, 2)).save(png, format='PNG')
        body = (b'--fixture\r\nContent-Disposition: form-data; name="photo"; filename="image.png"\r\n'
                b'Content-Type: image/png\r\n\r\n' + png.getvalue() +
                b'\r\n--fixture\r\nContent-Disposition: form-data; name="cam"; filename="cam.txt"\r\n\r\n'
                b'1 0 0\n0 1 0\n0 0 1\n\r\n--fixture--\r\n')
        httpd = make_server('127.0.0.1', 0, server.app, threaded=True)
        serving = threading.Thread(target=httpd.serve_forever); serving.start()
        responses, errors = [], []
        def request():
            try:
                req = urllib.request.Request('http://127.0.0.1:' + str(httpd.server_port) + '/upload', body,
                                             {'Content-Type': 'multipart/form-data; boundary=fixture'})
                with urllib.request.urlopen(req, timeout=8) as response:
                    responses.append(json.loads(gzip.decompress(response.read())))
            except Exception as error:
                errors.append(repr(error))
        workers = [threading.Thread(target=request) for _ in range(2)]
        try:
            with patch.object(server.Image, 'open', side_effect=open_image):
                for worker in workers: worker.start()
                try:
                    self.assertTrue(entered.wait(3))
                    self.assertTrue(both_decoded.wait(3))
                    self.assertEqual(len(state['calls']), 1)
                    # Health remains available while inference and its waiter are blocked.
                    with urllib.request.urlopen('http://127.0.0.1:' + str(httpd.server_port) + '/health', timeout=2) as response:
                        self.assertEqual(response.status, 200)
                finally:
                    release.set()
                for worker in workers: worker.join(8)
        finally:
            release.set()
            httpd.shutdown(); httpd.server_close(); serving.join()
        self.assertEqual(errors, [])
        self.assertEqual(responses, [{'fixture': True}, {'fixture': True}])
        self.assertEqual(state['initializations'], 1)
        self.assertEqual(state['peak'], 1)
        self.assertEqual(len(state['calls']), 2)
        self.assertEqual(state['calls'][0], state['calls'][1])


if __name__ == '__main__':
    unittest.main()
