import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from adadextrack.data.dataset import read_manifest
from adadextrack.data.groups import partition_groups
from adadextrack.replay_batch import partition_batches, run_groups, run_batches

ROOT = Path(__file__).resolve().parents[1]


class ReplayBatchTests(unittest.TestCase):
    def entries(self, size):
        return [SimpleNamespace(reference_id=str(i), group_id=str(i//2), object_name='stamp') for i in range(size)]

    def test_full_and_train_coverage(self):
        for filename, count in [('manifest.csv', 444), ('splits/unseen_trajectory_train.csv', 435),
                                 ('splits/unseen_object_train.csv', 402)]:
            entries = read_manifest(ROOT/'data/adadextrack_full'/filename)
            groups = partition_groups(entries)
            self.assertEqual(len(groups), count)
            self.assertEqual({e.reference_id for g in groups for e in g}, {e.reference_id for e in entries})
            self.assertTrue(all(len({e.group_id for e in g}) == 1 for g in groups))
            self.assertEqual(sum(map(len, groups)), len(entries))
        with self.assertRaises(ValueError):
            partition_groups(self.entries(1)*2)

    def test_reference_batches_cover_full_and_training_manifests_in_order(self):
        for filename, sizes in [('manifest.csv', [256]*10+[205]),
                                ('splits/unseen_trajectory_train.csv', [256]*8+[164])]:
            entries = read_manifest(ROOT/'data/adadextrack_full'/filename)
            batches = partition_batches(entries, 256)
            self.assertEqual(list(map(len, batches)), sizes)
            self.assertEqual([entry for batch in batches for entry in batch], entries)
            self.assertTrue(any(len({entry.group_id for entry in batch}) > 1 for batch in batches))
        with self.assertRaises(ValueError):
            partition_batches(self.entries(5), 0)
        with self.assertRaises(ValueError):
            partition_batches(self.entries(1)*2, 256)

    def test_batch_cli_preserves_manifest_order_and_defaults_to_256(self):
        from adadextrack.replay import main
        with patch('adadextrack.replay_batch.run_batches', return_value=0) as run:
            self.assertEqual(main(['--dataset','full','--all','--mode','batch','--overview']),0)
            args, entries = run.call_args.args
            self.assertEqual(args.batch_size, 256)
            self.assertTrue(args.overview)
            self.assertEqual([entry.reference_id for entry in entries], list(map(str,range(2765))))
            self.assertEqual(main(['--all','--mode','batch','--output','/tmp/example_replay.npz']),0)
            self.assertEqual(len(run.call_args.args[1]),7)
        with self.assertRaises(SystemExit):
            main(['--mode','batch','--batch-size','0'])

    def run_coordinator(self, directory, cancel_index=None, fail_index=None, mode='group'):
        root = Path(directory)
        manifest = root/'manifest.csv'
        manifest.write_text('test manifest\n')
        args = argparse.Namespace(manifest=manifest, output_dir=root/'run', output=None,
            loops=2, headless=True, overview=False, device='cuda:0', fps=0,
            snapshot_frame=150, snapshot=None, mode=mode, batch_size=3)
        calls = []

        def worker(command, **kwargs):
            config = json.loads(Path(command[-1]).read_text())
            index = len(calls)
            calls.append(config)
            code = 1 if fail_index == index else 0
            if code == 0:
                output = Path(config['output'])
                frames = 17 if cancel_index == index else 600
                refs = config['reference_id']
                np.savez_compressed(str(output), obj_poses=np.zeros((frames,len(refs),7)),
                    robot_qpos=np.zeros((frames,len(refs),22)), reference_ids=np.array(list(map(str,refs))),
                    reference_frames=np.arange(frames)%300)
                report = dict(status='cancelled' if cancel_index == index else 'complete',
                    references=len(refs), frames=frames, max_object_state_error=0, max_joint_error=0,
                    output=str(output), snapshot=None)
                output.with_suffix('.json').write_text(json.dumps(report))
            return SimpleNamespace(stdout=iter(['worker output\n']), wait=lambda:code)

        with patch('adadextrack.replay_batch.subprocess.Popen', side_effect=worker):
            runner = run_groups if mode == 'group' else run_batches
            if fail_index is not None:
                with self.assertRaises(RuntimeError):
                    runner(args, self.entries(5))
            else:
                self.assertEqual(runner(args, self.entries(5)), 0)
        return calls, json.loads((root/'run/summary.json').read_text())

    def test_each_worker_plays_only_one_group(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, summary = self.run_coordinator(directory)
            self.assertEqual([call['reference_id'] for call in calls], [[0,1],[2,3],[4]])
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['completed_references'], 5)
            self.assertEqual([call['group_id'] for call in calls], [0, 1, 2])
            self.assertEqual(summary['completed_groups'], 3)
            self.assertEqual(summary['expected_frames_per_reference'], 600)

    def test_closing_viewer_stops_remaining_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, summary = self.run_coordinator(directory, cancel_index=1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(summary['status'], 'cancelled')
            self.assertEqual(summary['completed_references'], 2)
            self.assertEqual(summary['completed_groups'], 1)
            self.assertEqual(summary['groups'][-1]['frames'], 17)

    def test_failure_does_not_skip_a_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, summary = self.run_coordinator(directory, fail_index=0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['completed_references'], 0)

    def test_batch_workers_receive_mixed_groups_and_last_partial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, summary = self.run_coordinator(directory, mode='batch')
            self.assertEqual([call['reference_id'] for call in calls], [[0,1,2],[3,4]])
            self.assertTrue(all(call['group_id'] is None and call['mode']=='batch' for call in calls))
            self.assertEqual(summary['playback_mode'], 'batch')
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['completed_references'], 5)
            self.assertEqual(summary['completed_batches'], 2)

    def test_closing_batch_viewer_stops_remaining_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, summary = self.run_coordinator(directory, cancel_index=0, mode='batch')
            self.assertEqual(len(calls),1)
            self.assertEqual(summary['status'],'cancelled')
            self.assertEqual(summary['completed_batches'],0)

    def test_batch_failure_stops_remaining_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, summary = self.run_coordinator(directory, fail_index=0, mode='batch')
            self.assertEqual(len(calls),1)
            self.assertEqual(summary['status'],'failed')

    def test_full_dataset_splits_reuse_global_ids(self):
        root = ROOT/'data/adadextrack_full'
        entries = read_manifest(root/'manifest.csv', verify_hashes=True)
        self.assertEqual([e.reference_id for e in entries], list(map(str,range(2765))))
        self.assertEqual(len({entry.object_name for entry in entries}), 50)
        for prefix, sizes in (('unseen_trajectory',(2212,553)), ('unseen_object',(2520,245))):
            subsets = [read_manifest(root/'splits'/(prefix+'_'+name+'.csv')) for name in ('train','test')]
            self.assertEqual(tuple(map(len,subsets)), sizes)
            ids = [{e.reference_id for e in subset} for subset in subsets]
            self.assertFalse(ids[0] & ids[1])
            self.assertEqual(ids[0] | ids[1], {e.reference_id for e in entries})
            by_id = {e.reference_id:e for e in entries}
            for subset in subsets:
                self.assertEqual(subset, [by_id[e.reference_id] for e in subset])


if __name__ == '__main__':
    unittest.main()
