import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_site', ROOT/'scripts/build_site.py')
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


class WebsiteTests(unittest.TestCase):
    def test_only_public_site_files_are_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'repository'
            root.mkdir()
            for name in ('index.html','manifest.json'):
                (root/name).write_text(name)
            for name in ('static','videos','thumbs','assets','data','checkpoints','logs','adadextrack'):
                (root/name).mkdir()
                (root/name/'example').write_text(name)
            output=site.build_site(root,root/'_site')
            self.assertEqual({p.name for p in output.iterdir()},
                {'index.html','manifest.json','.nojekyll','static','videos','thumbs'})
            self.assertEqual((output/'videos/example').read_text(),'videos')
            self.assertEqual((output/'index.html').read_text(),'index.html')
            with self.assertRaises(FileExistsError):
                site.build_site(root,output)

    def test_output_cannot_be_inside_a_published_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'index.html').touch()
            (root/'manifest.json').write_text('[]')
            with self.assertRaises(ValueError):
                site.build_site(root,root/'videos/recursive')
