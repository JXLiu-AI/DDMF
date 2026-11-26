import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('download_models', Path(__file__).resolve().parents[1] / 'scripts/download_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def test_download_and_skip_verified_file(self):
        with tempfile.TemporaryDirectory() as directory:
            models = {'sample': ('sample.bin', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad', 'models/sample.bin')}
            with mock.patch.object(module, 'MODELS', models), mock.patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(b'abc')) as network:
                module.download('sample', Path(directory))
                module.download('sample', Path(directory))
                self.assertEqual(network.call_count, 1)
            self.assertEqual((Path(directory) / 'models/sample.bin').read_bytes(), b'abc')

    def test_bad_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            models = {'sample': ('sample.bin', '0' * 64, 'models/sample.bin')}
            with mock.patch.object(module, 'MODELS', models), mock.patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(b'bad')):
                with self.assertRaises(ValueError):
                    module.download('sample', Path(directory))
            self.assertEqual(list((Path(directory) / 'models').iterdir()), [])

    def test_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test'
            path.write_bytes(b'abc')
            self.assertEqual(module.sha256(path), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            relative = module.MODELS['ios'][2]
            path = Path(directory) / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b'existing')
            with mock.patch.object(module.urllib.request, 'urlopen') as network:
                with self.assertRaises(ValueError):
                    module.download('ios', Path(directory))
                network.assert_not_called()
            self.assertEqual(path.read_bytes(), b'existing')


if __name__ == '__main__':
    unittest.main()
