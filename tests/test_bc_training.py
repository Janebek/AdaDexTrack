"""CPU-only tests of the collection/BC pipeline; no Isaac Gym or real training."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from adadextrack.data.groups import ROOT, SOURCES, select_groups
from adadextrack.inference import build_model, make_policy
from adadextrack.collection import (collection_configuration, file_digest, parse_args as collect_args,
                                   prepare_collection, rollout_round, save_round)
from adadextrack.bc import (action_loss, iter_batches, parse_args as bc_args, read_collection, train)
from adadextrack.specialists import run_jobs


def tiny_params():
    params = yaml.safe_load((ROOT / 'configs/bc_network.yaml').read_text())['params']
    params['network']['mlp']['units'] = [16, 8]
    return params


def make_teachers(root):
    for group in select_groups(SOURCES['examples'], all_groups=True):
        params = tiny_params()
        model = make_policy(params)
        metadata = dict(schema_version=1, policy_type='specialist', observation_dim=273, action_dim=22,
                        dataset='examples', group_id=int(group[0].group_id),
                        reference_ids=[e.reference_id for e in group], reference_sha256=[e.sha256 for e in group],
                        model=params['model'], network=params['network'])
        folder = Path(root) / 'examples' / ('group_%03d' % int(group[0].group_id))
        folder.mkdir(parents=True)
        torch.save(dict(model=model.state_dict(), adadextrack=metadata), str(folder/'best.pth'))


def make_data(folder, group_ids=None, rounds=1):
    root = Path(folder)
    teacher_root = root / 'teachers'
    make_teachers(teacher_root)
    output = root/'data'
    arguments = ['--specialist-root', str(teacher_root), '--output-dir', str(output),
                 '--envs-per-reference', '1', '--rounds', str(rounds)]
    for group_id in group_ids or []:
        arguments.extend(['--group-id', str(group_id)])
    args = collect_args(arguments)
    jobs, index = prepare_collection(args)
    output.mkdir()
    for job in jobs:
        Path(job['directory']).mkdir()
        rng = np.random.RandomState(job['group_id'])
        for round_index in range(rounds):
            observations = rng.normal(size=(299, job['num_envs'], 273)).astype(np.float32)
            # A simple target that a tiny student can learn with the actual BC objective.
            actions = np.full((299, job['num_envs'], 22), .2, dtype=np.float32)
            index['shards'].extend(save_round(job, round_index, observations, actions))
    index.update(status='complete', samples=sum(s['samples'] for s in index['shards']))
    (output/'index.json').write_text(json.dumps(index))
    return output, index


class CollectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_released_specialists_use_the_explicit_checkpoint_root(self):
        with tempfile.TemporaryDirectory() as folder:
            specialist_root = Path(folder) / 'checkpoints/specialists'
            make_teachers(specialist_root)
            args = collect_args(['--specialist-root', str(specialist_root)])
            jobs, index = prepare_collection(args)
            self.assertEqual(index['group_ids'], [322, 369, 435])
            self.assertEqual(len(index['references']), 7)
            self.assertEqual([job['num_envs'] for job in jobs], [40, 80, 20])
            for job in jobs:
                self.assertEqual(Path(job['checkpoint']), specialist_root / 'examples' /
                                 ('group_%03d' % job['group_id']) / 'best.pth')

    def test_defaults_selection_checkpoint_discovery_and_membership(self):
        self.assertEqual((collect_args([]).envs_per_reference, collect_args([]).rounds), (20, 10))
        self.assertEqual(bc_args([]).batch_size, 2000)
        for dataset, groups, count in [('examples', 3, 7), ('unseen_ref', 435, 2212), ('unseen_obj', 402, 2520)]:
            selected = select_groups(SOURCES[dataset], all_groups=True)
            self.assertEqual((len(selected), sum(map(len, selected))), (groups, count))
            args = collect_args(['--dataset', dataset, '--all'])
            self.assertEqual(args.output_dir, (ROOT/'outputs/bc_data'/dataset).resolve())
            self.assertEqual(bc_args(['--dataset', dataset]).data_dir, args.output_dir)
        with tempfile.TemporaryDirectory() as folder:
            make_teachers(folder)
            args = collect_args(['--specialist-root', folder])
            jobs, index = prepare_collection(args)
            self.assertEqual([j['group_id'] for j in jobs], [322, 369, 435])
            self.assertEqual([j['num_envs'] for j in jobs], [40, 80, 20])
            self.assertFalse(index['settings']['success_filter'])
            task = collection_configuration(jobs[0])
            self.assertEqual((task['env']['initial_object_xy_noise'], task['env']['initial_object_yaw_noise_degrees']), (.05, 30))
            self.assertTrue(task['task']['randomize'])
            self.assertFalse(task['env']['random_time'])
            self.assertFalse(task['env']['apply_lag'])
            checkpoint = Path(jobs[0]['checkpoint'])
            state = torch.load(str(checkpoint), weights_only=False)
            state['adadextrack']['reference_ids'] = ['1237']
            torch.save(state, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'membership'):
                prepare_collection(args)
            checkpoint.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing specialist'):
                prepare_collection(args)

    def test_reset_alignment_two_rounds_and_all_rollouts_retained(self):
        class Env:
            def __init__(self):
                self.num_envs = 2
                self.progress_buf = torch.zeros(2, dtype=torch.long)
                self.reset_buf = torch.ones(2, dtype=torch.bool)
                self.obs = torch.zeros(2, 273)
                self.seen = []
            def step(self, actions):
                self.seen.append(actions.clone())
                if self.reset_buf.all():
                    self.progress_buf.zero_()
                self.progress_buf.add_(1)
                self.reset_buf = self.progress_buf == 300
                self.obs.fill_(float(self.progress_buf[0]))
                # Deliberately awful rewards: collection must still retain every rollout.
                return {'obs': self.obs}, torch.full((2,), -100.), self.reset_buf, {}

        class Teacher(torch.nn.Module):
            def forward(self, inputs):
                return {'mus': inputs['obs'][:, :22] + 2.}

        env, model = Env(), Teacher()
        observation = env.obs
        for _ in range(2):
            obs, actions, observation = rollout_round(env, model, observation)
            self.assertEqual(obs.shape, (299, 2, 273))
            np.testing.assert_array_equal(obs[:, 0, 0], np.arange(1, 300))
            np.testing.assert_array_equal(actions, obs[:, :, :22] + 2.)
        self.assertTrue(all(bool((a == 1).all()) for a in env.seen))
        self.assertFalse(model.training)

    def test_full_partitions_discover_every_checkpoint_with_complete_members(self):
        # Exercise full-scale orchestration without constructing hundreds of networks.
        for dataset, count, references in [('unseen_ref', 435, 2212), ('unseen_obj', 402, 2520)]:
            with tempfile.TemporaryDirectory() as folder:
                groups = select_groups(SOURCES[dataset], all_groups=True)
                metadata = {}
                for group in groups:
                    group_id = int(group[0].group_id)
                    checkpoint = Path(folder)/dataset/('group_%03d' % group_id)/'best.pth'
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b'coordinator-test')
                    metadata[str(checkpoint)] = dict(policy_type='specialist', dataset=dataset, group_id=group_id,
                        reference_ids=[e.reference_id for e in group], reference_sha256=[e.sha256 for e in group])
                args = collect_args(['--dataset', dataset, '--all', '--specialist-root', folder])
                def load(path):
                    return SimpleNamespace(adadextrack_metadata=metadata[str(path)])
                with patch('adadextrack.inference.build_model', side_effect=load), \
                     patch('adadextrack.data.reference.load_reference'):
                    jobs, index = prepare_collection(args)
                self.assertEqual(len(jobs), count)
                self.assertEqual(len(index['references']), references)
                self.assertEqual(sum(len(j['reference_ids']) for j in jobs), references)
                self.assertTrue(all(len(j['reference_ids'])*20 == j['num_envs'] for j in jobs))
                self.assertTrue(all(j['dataset'] == dataset for j in jobs))

    def test_collection_uses_shared_queue_with_arbitrary_gpu_list(self):
        # Existing scheduler tests cover single/multiple GPUs, failures and cleanup;
        # this verifies the collection worker is selected rather than the PPO worker.
        with tempfile.TemporaryDirectory() as folder:
            job = dict(directory=folder, group_id=322, reference_ids=['1237'])
            process = unittest.mock.Mock()
            process.poll.return_value = 0
            process.wait.return_value = 0
            def launch(command, **kwargs):
                self.assertEqual(Path(command[-3]).name, 'collect_specialists.py')
                (Path(folder)/'status.json').write_text(json.dumps({'status': 'complete'}))
                return process
            with patch('adadextrack.specialists.subprocess.Popen', side_effect=launch):
                self.assertEqual(run_jobs([job], list(map(str, range(8))),
                                          worker_script=ROOT/'scripts/collect_specialists.py', label='COLLECT'), 0)


class BCDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_shard_coverage_metadata_cache_and_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, index = make_data(folder, rounds=2)
            loaded, store, digest = read_collection(directory, 'examples', cache_mb=1)
            self.assertEqual(loaded['samples'], 7*2*299)
            self.assertEqual(len(loaded['shards']), 14)
            self.assertEqual(len(digest), 64)
            batches = list(iter_batches(index['shards'], store, 103, 42))
            self.assertEqual(sum(len(o) for o, _ in batches), index['samples'])
            self.assertTrue(all(len(o) >= 2 for o, _ in batches))
            self.assertLessEqual(store.used, store.limit)
            for shard in index['shards']:
                with np.load(directory/shard['path']) as data:
                    self.assertEqual(set(data['frame']), set(range(299)))
            again = list(iter_batches(index['shards'], store, 103, 42))
            self.assertTrue(all(torch.equal(a[0], b[0]) for a, b in zip(batches, again)))
            # A final singleton is merged, rather than dropped or passed to unbiased RMS variance.
            one = [index['shards'][0]]
            self.assertEqual([len(o) for o, _ in iter_batches(one, store, 149, 1)], [149, 150])
            damaged = copy.deepcopy(index)
            damaged['shards'].pop()
            (directory/'index.json').write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                read_collection(directory, 'examples')
            (directory/'index.json').write_text(json.dumps(index))
            with (directory/index['shards'][0]['path']).open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_collection(directory, 'examples')

    def test_loss_clamps_targets_and_sums_action_dimensions(self):
        predicted = torch.zeros(3, 22, requires_grad=True)
        target = torch.full_like(predicted, 2.)
        loss = action_loss(predicted, target)
        self.assertEqual(float(loss), 22.)
        loss.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertEqual(float(action_loss(torch.ones(3, 22), target)), 0.)

    def test_real_bc_fitting_checkpoint_inference_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data_dir, index = make_data(root, group_ids=[435])
            params = tiny_params()
            config = root/'network.yaml'
            config.write_text(yaml.safe_dump({'params': params}))
            def arguments(output, epochs, resume=False):
                values = ['--data-dir', str(data_dir), '--output-dir', str(output), '--network-config', str(config),
                          '--device', 'cpu', '--epochs', str(epochs), '--batch-size', '64', '--learning-rate', '.005']
                return bc_args(values + (['--resume'] if resume else []))
            continuous, resumed = root/'continuous', root/'resumed'
            train(arguments(continuous, 8))
            train(arguments(resumed, 3))
            train(arguments(resumed, 8, True))
            a = torch.load(str(continuous/'last.pth'), weights_only=False)
            b = torch.load(str(resumed/'last.pth'), weights_only=False)
            self.assertEqual(a['epoch'], 8)
            self.assertTrue(all(torch.equal(value, b['model'][key]) for key, value in a['model'].items()))
            losses = [json.loads(line)['action_mse'] for line in (continuous/'metrics.jsonl').read_text().splitlines()]
            self.assertLess(losses[-1], losses[0] * .2)
            self.assertGreater(float(a['model']['running_mean_std.count']), 1.)
            loaded = build_model(continuous/'best.pth')
            self.assertEqual(loaded.adadextrack_metadata['policy_type'], 'bc_tracker')
            self.assertEqual(loaded.adadextrack_metadata['group_ids'], [435])
            stats = {k: v.clone() for k, v in loaded.state_dict().items() if k.startswith('running_mean_std.')}
            with torch.no_grad():
                result = loaded(dict(is_train=False, obs=torch.ones(2, 273), prev_actions=None, rnn_states=None))
            self.assertEqual(result['mus'].shape, (2, 22))
            self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in stats.items()))
            from adadextrack.inference import parse_args, preflight
            inference_args = parse_args(['--checkpoint', str(continuous/'best.pth'), '--ref-csv', str(SOURCES['examples'])])
            # A BC tracker can evaluate other groups; the specialist-only restriction must not apply.
            references, _ = preflight(inference_args)
            self.assertEqual(len(references), 7)
            bad_args = arguments(resumed, 9, True)
            bad_args.seed = 100
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                train(bad_args)
            self.assertNotIn('isaacgym', __import__('sys').modules)


if __name__ == '__main__':
    unittest.main()
