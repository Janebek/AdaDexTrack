"""CPU checks for the released AE contract and reproducible sampling."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from adadextrack import object_ae as ae
from adadextrack.models.object_ae import PointAE, chamfer_l2


class ObjectAETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_chamfer_chunk_gradient_and_known_distance(self):
        pred = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]], requires_grad=True)
        target = torch.tensor([[[1., 0., 0.], [3., 0., 0.], [5., 0., 0.]]])
        full = chamfer_l2(pred, target)
        self.assertAlmostEqual(full.item(), 1 + 11/3, places=5)
        gradient = torch.autograd.grad(full, pred)[0]
        chunked = chamfer_l2(pred, target, chunk=1)
        torch.testing.assert_close(chunked, full)
        torch.testing.assert_close(torch.autograd.grad(chunked, pred)[0], gradient)

    def test_recovered_split_and_original_weights(self):
        manifest = json.loads((ae.ROOT/'data/object_ae/manifest.json').read_text())
        records = manifest['files']
        self.assertEqual(len(records), 57)
        train, val = ae.split_indices(57)
        self.assertEqual({records[i]['object_name'] for i in val}, {'flute', 'rubberduck'})
        self.assertEqual(len(train), 55)
        path = ae.ROOT/'checkpoints/object_ae.pth'
        if not path.exists():
            self.skipTest('Optional released AE weights not installed')
        self.assertEqual(ae.digest(path), manifest['original_checkpoint_sha256'])
        model, state = ae.load_model(path, 'cpu')
        self.assertEqual(state['epoch'], 926)
        self.assertIn('optim', state)
        self.assertEqual(model.encoder.fc_mu.out_features, 64)
        latent = torch.from_numpy(np.load(str(next((ae.ROOT/'assets/latents').glob('*.npy')))))[None]
        with torch.no_grad():
            decoded = model.decoder(latent)
        self.assertEqual(decoded.shape, (1,2048,3))
        self.assertTrue(torch.isfinite(decoded).all())

    def test_explicit_sampling_encoding_and_resume(self):
        trimesh = ae.trimesh_module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root/'meshes'; inputs.mkdir()
            for index in range(4):
                mesh = trimesh.creation.icosphere(subdivisions=0, radius=1+index/10)
                mesh.export(str(inputs/('shape%d.ply'%index)))
            first = ae.sample_points(inputs/'shape0.ply', 16, 42)[0]
            second = ae.sample_points(inputs/'shape0.ply', 16, 42)[0]
            np.testing.assert_array_equal(first, second)
            self.assertFalse(np.array_equal(first, ae.sample_points(inputs/'shape0.ply',16,43)[0]))
            args = ae.parse_args(['train','--input',str(inputs),'--output-dir',str(root/'full'),
                                 '--epochs','2','--num-points','8','--latent-dim','4','--batch-size','2',
                                 '--val-split','.25','--workers','0','--device','cpu'])
            ae.train(args)
            full = torch.load(str(root/'full/last.pth'),weights_only=False)
            saved = ae.save_checkpoint
            def interrupt(path, state):
                saved(path, state)
                if path.name == 'last.pth' and state['epoch'] == 2:
                    raise KeyboardInterrupt
            args.output_dir = root/'resumed'
            with patch.object(ae, 'save_checkpoint', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    ae.train(args)
            args.resume = args.output_dir/'last.pth'
            ae.train(args)
            resumed = torch.load(str(args.output_dir/'last.pth'), weights_only=False)
            self.assertEqual(resumed['epoch'], 3)
            for key, value in full['model'].items():
                torch.testing.assert_close(value, resumed['model'][key], rtol=0, atol=0)
            self.assertEqual(full['sched'], resumed['sched'])
            args.output_dir = root/'copied_resume'
            args.resume = root/'full/last.pth'
            ae.train(args)
            self.assertTrue((args.output_dir/'best.pth').is_file())
            self.assertTrue((args.output_dir/'last.pth').is_file())
            torch.testing.assert_close(torch.get_rng_state(),full['rng']['torch'],rtol=0,atol=0)
            for name in ('a','b'):
                encoding = ae.parse_args(['encode','--checkpoint',str(args.output_dir/'last.pth'),
                                         '--input',str(inputs),'--output-dir',str(root/name),'--device','cpu'])
                ae.encode(encoding)
            for path in (root/'a').glob('*.npy'):
                self.assertEqual(path.read_bytes(), (root/'b'/path.name).read_bytes())
            with self.assertRaises(FileExistsError):
                ae.encode(encoding)
            ae.decode(ae.parse_args(['decode','--checkpoint',str(args.output_dir/'last.pth'),
                      '--latent-dir',str(root/'a'),'--output-dir',str(root/'decoded'),'--device','cpu']))
            self.assertEqual(len(list((root/'decoded').glob('*.ply'))),4)


if __name__ == '__main__':
    unittest.main()
