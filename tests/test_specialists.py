import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adadextrack.data.groups import ROOT, SOURCES, select_groups
from adadextrack.specialists import gpu_tokens, parse_args, prepare_jobs, run_jobs, training_configuration


class SpecialistSelectionTests(unittest.TestCase):
    def test_fixed_examples_and_split_membership(self):
        examples = select_groups(SOURCES['examples'], all_groups=True, verify_hashes=True)
        self.assertEqual([[int(e.reference_id) for e in g] for g in examples],
                         [[1237, 1232], [1094, 1089, 1087, 1093], [2099]])
        self.assertEqual([int(g[0].group_id) for g in examples], [322, 369, 435])
        debug = select_groups(SOURCES['examples'], reference_ids=[2099, 1237])
        self.assertEqual([[e.reference_id for e in g] for g in debug], [['1237'], ['2099']])
        training = {e.reference_id for g in select_groups(SOURCES['unseen_ref'], all_groups=True) for e in g}
        self.assertTrue({e.reference_id for g in examples for e in g} <= training)
        for source, count, references in [('unseen_ref', 435, 2212), ('unseen_obj', 402, 2520)]:
            groups = select_groups(SOURCES[source], all_groups=True)
            self.assertEqual(len(groups), count)
            self.assertEqual(sum(map(len, groups)), references)
            self.assertTrue(all(len({e.group_id for e in g}) == 1 for g in groups))
        self.assertEqual(len(select_groups(SOURCES['unseen_ref'], [322])[0]), 9)
        with self.assertRaises(ValueError):
            select_groups(SOURCES['unseen_ref'], [50])  # this group belongs only to test

    def test_defaults_and_gpu_visibility(self):
        args = parse_args([])
        self.assertEqual((args.dataset, args.num_envs, args.epochs, args.seed), ('examples', 8192, 2001, 42))
        self.assertEqual(gpu_tokens(environ={}), ['0'])
        self.assertEqual(gpu_tokens(environ={'CUDA_VISIBLE_DEVICES': '3,5'}), ['3'])
        self.assertEqual(gpu_tokens([1, 0], {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b'}), ['GPU-b', 'GPU-a'])
        for indices, env in [([0, 0], {}), ([2], {'CUDA_VISIBLE_DEVICES': '3,5'}), (None, {'CUDA_VISIBLE_DEVICES': ''})]:
            with self.assertRaises(ValueError):
                gpu_tokens(indices, env)

    def test_configuration_and_resume_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parse_args(['--output-dir', folder])
            jobs = prepare_jobs(args)
            self.assertEqual([job['group_id'] for job in jobs], [322, 369, 435])
            task, params = training_configuration(jobs[0])
            self.assertEqual(task['env']['reference_ids'], ['1237', '1232'])
            self.assertEqual(task['env']['initial_object_xy_noise'], .05)
            self.assertEqual(task['env']['initial_object_yaw_noise_degrees'], 30)
            self.assertTrue(task['task']['randomize'])
            self.assertTrue(task['env']['random_time'])
            self.assertEqual(params['network']['mlp']['units'], [1024, 512, 256, 128])
            self.assertEqual(params['config']['max_epochs'], 2001)
            self.assertFalse(params['config']['multi_gpu'])
            self.assertNotIn('score_to_win', params['config'])
            for job in jobs:
                directory = Path(job['directory'])
                directory.mkdir(parents=True)
                (directory/'job.json').write_text(json.dumps(job))
                (directory/'status.json').write_text(json.dumps({'status': 'complete', 'epoch': 2001}))
                (directory/'last.pth').touch()
                (directory/'best.pth').touch()
            self.assertEqual(prepare_jobs(args), [])
            status_path = Path(jobs[0]['directory'])/'status.json'
            status_path.write_text(json.dumps({'status': 'failed'}))
            with self.assertRaisesRegex(ValueError, 'resume'):
                prepare_jobs(args)
            args.resume = True
            resumed = prepare_jobs(args)
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed[0]['resume_from'], str(Path(jobs[0]['directory'])/'last.pth'))
            args.seed = 7
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                prepare_jobs(args)


class SchedulerTests(unittest.TestCase):
    def run_queue(self, gpus, fail=None):
        calls, processes = [], []
        with tempfile.TemporaryDirectory() as folder:
            jobs = [dict(group_id=group, reference_ids=[str(group)], directory=str(Path(folder)/str(group)), epochs=2)
                    for group in [322, 369, 435]]

            class Process:
                def __init__(self, job, gpu):
                    self.job, self.gpu, self.ticks, self.done = job, gpu, 0, False
                def poll(self):
                    self.ticks += 1
                    if self.ticks == 1:
                        return None
                    self.done = True
                    code = int(self.job['group_id'] == fail)
                    (Path(self.job['directory'])/'status.json').write_text(json.dumps(
                        {'status': 'failed' if code else 'complete', 'epoch': 2}))
                    return code
                def wait(self, timeout=None):
                    return int(self.job['group_id'] == fail)

            def launch(command, **kwargs):
                job = json.loads(Path(command[-1]).read_text())
                gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
                self.assertTrue(all(p.done or p.gpu != gpu for p in processes))
                self.assertLess(sum(not p.done for p in processes), len(gpus))
                self.assertTrue(kwargs['start_new_session'])
                process = Process(job, gpu)
                calls.append(job['group_id'])
                processes.append(process)
                return process

            with patch('adadextrack.specialists.subprocess.Popen', side_effect=launch), patch('adadextrack.specialists.time.sleep'):
                result = run_jobs(jobs, gpus)
            statuses = [json.loads((Path(j['directory'])/'status.json').read_text())['status'] for j in jobs]
        return calls, result, statuses

    def test_single_gpu_runs_groups_sequentially(self):
        calls, result, states = self.run_queue(['4'])
        self.assertEqual(calls, [322, 369, 435])
        self.assertEqual(result, 0)
        self.assertEqual(states, ['complete']*3)

    def test_multiple_gpus_and_independent_failure(self):
        calls, result, states = self.run_queue(['0', '1'], fail=369)
        self.assertEqual(calls, [322, 369, 435])
        self.assertEqual(result, 1)
        self.assertEqual(states, ['complete', 'failed', 'complete'])

    def test_interrupt_terminates_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            job = dict(group_id=322, reference_ids=['1237'], directory=folder, epochs=2)
            process = SimpleNamespace(pid=12345, poll=lambda: None, wait=lambda timeout=None: 0)
            with patch('adadextrack.specialists.subprocess.Popen', return_value=process), \
                 patch('adadextrack.specialists.time.sleep', side_effect=KeyboardInterrupt), \
                 patch('adadextrack.specialists.os.killpg') as kill:
                with self.assertRaises(KeyboardInterrupt):
                    run_jobs([job], ['0'])
                kill.assert_called_once()
            self.assertEqual(json.loads((Path(folder)/'status.json').read_text())['status'], 'interrupted')


class CheckpointIntegrationTests(unittest.TestCase):
    def test_cpu_dummy_env_epoch_stop_resume_and_inference(self):
        # A tiny CPU-only synthetic environment exercises the real PPO adapter.
        # No Isaac Gym import, simulator or GPU work occurs in this test.
        import gym
        import numpy as np
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, str(ROOT/'third_party'))
        from rl_games.common import tr_helpers
        from rl_games.common.algo_observer import DefaultAlgoObserver
        from adadextrack.specialist_agent import SpecialistAgent
        from adadextrack.inference import build_model

        class DummyEnv:
            def __init__(self):
                self.step_count = 0
            def reset(self):
                return {'obs': torch.zeros(2, 273)}
            def step(self, actions):
                self.step_count += 1
                return {'obs': torch.ones(2, 273)*.1}, torch.ones(2), torch.ones(2, dtype=torch.bool), {}
            def get_env_state(self):
                return None
            def set_env_state(self, state):
                pass
            def set_train_info(self, *args):
                pass

        with tempfile.TemporaryDirectory() as folder:
            job = dict(directory=folder, num_envs=2, epochs=2, seed=42, group_id=322,
                       manifest=str(SOURCES['examples']), reference_ids=['1237', '1232'])
            _, params = training_configuration(job)
            metadata = dict(schema_version=1, policy_type='specialist', observation_dim=273, action_dim=22,
                            group_id=322, reference_ids=job['reference_ids'], signature='test-job',
                            model=copy.deepcopy(params['model']), network=copy.deepcopy(params['network']))

            def make_agent(epochs, directory=folder):
                _, params = training_configuration(dict(job, epochs=epochs, directory=directory))
                cfg = params['config']
                cfg.update(device='cpu', mini_epochs=1, save_best_after=1, save_frequency=1,
                           print_stats=False, vec_env=DummyEnv(), features={'observer': DefaultAlgoObserver()},
                           env_info={'observation_space': gym.spaces.Box(-np.inf, np.inf, shape=(273,)),
                                     'action_space': gym.spaces.Box(-1., 1., shape=(22,))})
                cfg['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**cfg['reward_shaper'])
                return SpecialistAgent(params, metadata, directory)

            agent = make_agent(2)
            try:
                self.assertEqual(agent.train()[1], 2)
                self.assertEqual(agent.vec_env.step_count, 16)
                self.assertTrue((Path(folder)/'best.pth').is_file())
                original = torch.load(str(Path(folder)/'last.pth'), map_location='cpu', weights_only=False)
                self.assertEqual(original['epoch'], 2)
                self.assertTrue(original['optimizer']['state'])
                self.assertGreater(original['model']['running_mean_std.count'].item(), 1)
                model = build_model(Path(folder)/'last.pth')
                with torch.no_grad():
                    output = model({'is_train': False, 'prev_actions': None, 'obs': torch.zeros(2,273), 'rnn_states': None})
                self.assertEqual(output['mus'].shape, (2,22))
                self.assertTrue(torch.isfinite(output['mus']).all())
                agent.last_mean_rewards = 1000.
                agent.save('last')
                copied_directory = Path(folder)/'continued'
                copied_directory.mkdir()
                copied = make_agent(3, str(copied_directory))
                try:
                    copied.restore(Path(folder)/'last.pth')
                    self.assertEqual((copied_directory/'best.pth').read_bytes(), (Path(folder)/'best.pth').read_bytes())
                finally:
                    copied.writer.close()
                resumed = make_agent(3)
                try:
                    resumed.restore(Path(folder)/'last.pth')
                    self.assertEqual(resumed.epoch_num, 2)
                    self.assertEqual(resumed.frame, agent.frame)
                    self.assertEqual(resumed.last_lr, agent.last_lr)
                    for key, value in original['optimizer']['state'].items():
                        torch.testing.assert_close(resumed.optimizer.state_dict()['state'][key]['exp_avg'], value['exp_avg'])
                    torch.testing.assert_close(resumed.model.running_mean_std.running_mean,
                                               agent.model.running_mean_std.running_mean)
                    resumed.train()
                    self.assertEqual(resumed.epoch_num, 3)
                    self.assertEqual(resumed.vec_env.step_count, 8)
                    self.assertEqual(resumed.last_mean_rewards, 1000.)
                    resumed.metadata['signature'] = 'wrong-group'
                    with self.assertRaisesRegex(ValueError, 'does not match'):
                        resumed.restore(Path(folder)/'last.pth')
                finally:
                    resumed.writer.close()
            finally:
                agent.writer.close()


if __name__ == '__main__':
    unittest.main()
