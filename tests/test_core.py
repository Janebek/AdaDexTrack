import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from adadextrack.data.dataset import read_manifest
from adadextrack.evaluation.metrics import compute_paper_metrics, aggregate_metrics, ERROR_FIELDS

ROOT=Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def targets(self):
        poses=np.zeros((300,7));poses[:,6]=1
        return dict(object_pose=poses.copy(),hand_pose=poses.copy(),qpos=np.zeros((300,22)))

    def score(self,actual):
        return compute_paper_metrics(actual['object_pose'],actual['hand_pose'],actual['qpos'],self.targets())

    def test_zero_error_and_quaternion_sign(self):
        actual=self.targets()
        actual['object_pose'][:,3:]*=-1
        result=self.score(actual)
        self.assertTrue(all(result[field]==0 for field in ERROR_FIELDS))
        self.assertEqual(result['completion_rate'],1)

    def test_finger_mae(self):
        actual=self.targets()
        actual['qpos'][:,:6]=10
        actual['qpos'][:,6:]=.1
        self.assertAlmostEqual(self.score(actual)['finger_mae_rad'],.1)

    def test_completion_differs_from_mean_success(self):
        actual=self.targets()
        actual['object_pose'][150:,0]=.36
        result=self.score(actual)
        self.assertTrue(result['mean_error_success'])
        self.assertEqual(result['completion_rate'],.5)

    def test_incomplete_or_nonfinite_rollout_rejected(self):
        actual=self.targets()
        with self.assertRaises(ValueError):
            compute_paper_metrics(actual['object_pose'][:-1],actual['hand_pose'],actual['qpos'],actual)
        actual['qpos'][0,6]=np.nan
        with self.assertRaises(ValueError):
            self.score(actual)

    def test_duplicate_or_missing_coverage_rejected(self):
        row=dict(reference_id='0',**self.score(self.targets()))
        for items,expected in [([row],['0','1']),([row,row],['0'])]:
            with self.assertRaises(ValueError):
                aggregate_metrics(items,expected)


class CheckpointStateTests(unittest.TestCase):
    def test_raw_and_converted_statistics_keep_the_same_values(self):
        from adadextrack.inference import canonical_model_state
        for prefix in ("running_mean_std.running_mean_std.obs.", "running_mean_std."):
            values = {name: object() for name in ("running_mean", "running_var", "count")}
            weight = object()
            state = {prefix + name: value for name, value in values.items()}
            state["a2c_network.mu.weight"] = weight
            normalized = canonical_model_state(state)
            for name, value in values.items():
                self.assertIs(normalized["running_mean_std." + name], value)
            self.assertIs(normalized["a2c_network.mu.weight"], weight)
            self.assertEqual(len(normalized), 4)

    def test_ambiguous_statistics_are_rejected(self):
        from adadextrack.inference import canonical_model_state
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            canonical_model_state({
                "running_mean_std.running_mean_std.obs.count": 1,
                "running_mean_std.count": 2,
            })


class DatasetTests(unittest.TestCase):
    def test_ids_hashes_and_groups(self):
        entries=read_manifest(ROOT/'data/adadextrack_full/manifest.csv',verify_hashes=True)
        self.assertEqual([entry.reference_id for entry in entries],list(map(str,range(2765))))
        self.assertEqual({entry.group_id for entry in entries},set(map(str,range(444))))
        self.assertEqual(len({entry.object_name for entry in entries}),50)
        members=[]
        for group_id in range(444):
            group=read_manifest(ROOT/'data/adadextrack_full/groups'/('group_{:03d}.csv'.format(group_id)))
            self.assertTrue(all(entry.group_id==str(group_id) for entry in group))
            members.extend(group)
        self.assertEqual(sorted(members,key=lambda entry:int(entry.reference_id)),entries)

    def test_entrypoints_default_to_full_dataset(self):
        from adadextrack.inference import parse_args
        from adadextrack.replay import main
        import yaml
        manifest=ROOT/'data/adadextrack_full/manifest.csv'
        args=parse_args([])
        self.assertEqual(args.checkpoint,ROOT/"checkpoints/bc_tracker_unseen_ref.pth")
        self.assertEqual(args.ref_csv,manifest)
        self.assertEqual(args.ref_base_dir,manifest.parent/'references')
        self.assertEqual(args.batch_size,256)
        cfg=yaml.safe_load((ROOT/'configs/tracking.yaml').read_text())
        self.assertEqual(ROOT/cfg['env']['reference_manifest'],manifest)
        with patch('adadextrack.replay_batch.run_groups',return_value=0) as run:
            self.assertEqual(main(['--all','--headless']),0)
            args,entries=run.call_args.args
            self.assertEqual(len(entries),7)
            self.assertEqual([entry.group_id for entry in entries],['322']*2+['369']*4+['435'])
            self.assertEqual(main(['--headless']),0)
            self.assertEqual(len(run.call_args.args[1]),2)
            self.assertEqual(main(['--dataset','full','--all','--headless']),0)
            args,entries=run.call_args.args
            self.assertEqual(args.manifest,manifest)
            self.assertEqual(len(entries),2765)
        with self.assertRaises(SystemExit):
            main(['--batch-size','256'])

    def test_small_scale_groups_have_different_objects(self):
        config=json.loads((ROOT/'configs/small_scale_bc.json').read_text())
        self.assertEqual(config['dataset'],'data/adadextrack_full/examples/specialists.csv')
        self.assertEqual(config['group_ids'],[322,369,435])
        self.assertEqual(config['references'],7)
        self.assertEqual(len({group['object_name'] for group in config['groups']}),3)
        for row in config['groups']:
            entries=read_manifest(ROOT/config['dataset'],reference_ids=row['reference_ids'])
            self.assertEqual(len(entries),int(row['reference_count']))
            self.assertTrue(all(entry.group_id==row['group_id'] for entry in entries))
            self.assertTrue(all(entry.group_id==row['group_id'] and entry.object_name==row['object_name'] for entry in entries))


if __name__=='__main__':
    unittest.main()
