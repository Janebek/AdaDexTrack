from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adadextrack.evaluation import evaluate


class ManifestResolutionTests(unittest.TestCase):
    def test_relocated_project_preserves_split_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            relative=Path('data/adadextrack_full/splits/unseen_trajectory_test.csv')
            manifest=root/relative
            manifest.parent.mkdir(parents=True)
            manifest.write_text('test split\n')
            old='/former/project/'+str(relative)
            with patch.object(evaluate,'ROOT',root):
                for settings in ({'reference_manifest':old},
                                 {'reference_manifest':old,'dataset_manifest_relative':str(relative)}):
                    self.assertEqual(evaluate.resolve_run_manifest(settings),manifest)

    def test_missing_split_cannot_fall_back_to_full_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            full=root/'data/adadextrack_full/manifest.csv'
            full.parent.mkdir(parents=True)
            full.write_text('full dataset\n')
            with patch.object(evaluate,'ROOT',root):
                with self.assertRaises(FileNotFoundError):
                    evaluate.resolve_run_manifest({'reference_manifest':
                        '/former/project/data/adadextrack_full/splits/missing.csv'})


if __name__=='__main__':
    unittest.main()
