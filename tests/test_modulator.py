"""CPU-only modulation timing, frozen tracker, and real PPO resume checks."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import gym
import numpy as np
import torch
from adadextrack.envs.modulator import ModulatorEnv, modulate_observation
from adadextrack.inference import ROOT, build_model
from adadextrack.modulator_training import configuration, parse_args
from adadextrack.data.latents import check_latent_metadata, latent_records


class FakeEnv:
    device = 'cpu'
    num_envs = 2
    clip_obs = 5.
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(273,))
    shadow_hand_dof_lower_limits = torch.linspace(-1.,0.,22)
    shadow_hand_dof_upper_limits = torch.linspace(1.,3.,22)
    def __init__(self):
        self.obs_buf = torch.zeros(2,273)
        self.step_count = 0
    def reset(self):
        return {'obs': self.obs_buf.clone()}
    def step(self, action):
        assert action.shape == (2,22)
        self.control = action.clone()
        self.step_count += 1
        self.obs_buf = torch.ones(2,273)*.1
        return {'obs': self.obs_buf.clone()}, torch.ones(2), torch.ones(2,dtype=torch.bool), {}
    def get_env_state(self):
        return None
    def set_env_state(self, state):
        pass
    def set_train_info(self, *args):
        pass


class FakeTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(22))
        self.register_buffer('count',torch.tensor(1.))
        self.observations = []
    def forward(self, data):
        if self.training:
            self.count += 1
        self.observations.append(data['obs'].clone())
        return {'mus': data['obs'][:,209:231]*self.weight}


class ModulatorTests(unittest.TestCase):
    def test_legacy_math_and_cached_timing(self):
        torch.manual_seed(42)
        env = FakeEnv()
        obs, action = torch.randn(2,273), torch.rand(2,202)*2-1
        result = modulate_observation(obs, action, env.shadow_hand_dof_lower_limits, env.shadow_hand_dof_upper_limits)
        expected = obs.clone()
        expected[:,71:99] = obs[:,71:99]+action[:,:28]*.1*.10
        for frame in range(4):
            section = slice(121+frame*22,143+frame*22)
            lower, upper = env.shadow_hand_dof_lower_limits, env.shadow_hand_dof_upper_limits
            physical = lower + (upper-lower)*(obs[:,section]+1)/2
            physical = physical + action[:,28+frame*22:50+frame*22]*.1*.314
            expected[:,section] = 2*(physical-lower)/(upper-lower)-1
        expected[:,209:273] = action[:,116:180]*.6
        torch.testing.assert_close(result,expected,rtol=1e-5,atol=3e-7)
        tracker = FakeTracker()
        wrapped = ModulatorEnv(env,tracker)
        wrapped.reset()
        wrapped.step(action)
        torch.testing.assert_close(env.control,action[:,180:202])
        torch.testing.assert_close(tracker.observations[0],torch.zeros(2,273))
        cached = wrapped.tracker_observation.clone()
        wrapped.step(torch.zeros(2,202))
        torch.testing.assert_close(tracker.observations[1],cached)
        torch.testing.assert_close(env.control,cached[:,209:231])
        self.assertEqual(tracker.count.item(),1)
        self.assertFalse(tracker.training)
        self.assertTrue(all(not p.requires_grad for p in tracker.parameters()))
        self.assertEqual(wrapped.num_acts,202)

    def test_latent_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            np.save(str(root/'Cup_latent.npy'),np.zeros(64,dtype=np.float32))
            hashes=latent_records(['cup'],root)
            self.assertEqual(check_latent_metadata({'latents':hashes},['cup'],root),hashes)
            np.save(str(root/'unseen_latent.npy'),np.zeros(64,dtype=np.float32))
            np.save(str(root/'Cup_latent.npy'),np.ones(64,dtype=np.float32))
            with self.assertRaisesRegex(ValueError,'different object latent'):
                check_latent_metadata({'latents':hashes},['unseen'],root)

    def test_inference_default_pairs_splits_and_custom_checkpoint(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("modulator_entry_test", ROOT/'scripts/infer_modulator.py')
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        for dataset, stem in [('unseen_ref','unseen_trajectory'),('unseen_obj','unseen_object')]:
            args = entry.parse_args(['--dataset',dataset])
            self.assertEqual(args.checkpoint,ROOT/('checkpoints/modulator_'+dataset+'.pth'))
            self.assertEqual(args.tracker_checkpoint,ROOT/('checkpoints/bc_tracker_'+dataset+'.pth'))
            self.assertEqual(args.ref_csv.name,stem+'_test.csv')
            self.assertEqual(entry.parse_args(['--dataset',dataset,'--split','train']).ref_csv.name,
                             stem+'_train.csv')
        example = entry.parse_args(['--dataset','examples','--render'])
        self.assertEqual(example.checkpoint,ROOT/'outputs/modulator/examples/best.pth')
        self.assertEqual(example.tracker_checkpoint,ROOT/'outputs/bc/examples/best.pth')
        self.assertTrue(example.render)
        custom = entry.parse_args(['--checkpoint','outputs/custom/best.pth','--tracker-checkpoint',
                                  'outputs/custom/bc.pth','--ref-csv','data/adadextrack_full/manifest.csv'])
        self.assertEqual(custom.checkpoint,(ROOT/'outputs/custom/best.pth').resolve())
        self.assertEqual(custom.tracker_checkpoint,(ROOT/'outputs/custom/bc.pth').resolve())
        self.assertEqual(custom.ref_csv,ROOT/'data/adadextrack_full/manifest.csv')

    def test_released_legacy_modulators_and_unknown_checksum(self):
        import json
        from adadextrack.inference import canonical_model_state, released_modulator_parameters
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'unknown.pth'
            path.write_bytes(b'not a released checkpoint')
            with self.assertRaisesRegex(ValueError,'Unrecognized legacy'):
                released_modulator_parameters(path)
        registry = json.loads((ROOT/'checkpoints/checkpoints.json').read_text())
        installed = 0
        for name in ('modulator_unseen_ref','modulator_unseen_obj'):
            path = ROOT/'checkpoints'/registry[name]['file']
            if not path.is_file():
                continue
            installed += 1
            model = build_model(path,action_dim=202)
            self.assertEqual(model.adadextrack_metadata['dataset'],registry[name]['dataset'])
            self.assertEqual(model.adadextrack_metadata['tracker_sha256'],registry[name]['tracker_sha256'])
            state = canonical_model_state(torch.load(str(path),map_location='cpu',weights_only=False)['model'])
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,state[key],rtol=0,atol=0)
            counts = {k:v.clone() for k,v in model.state_dict().items() if k.startswith('running_mean_std.')}
            with torch.no_grad():
                output=model(dict(is_train=False,prev_actions=None,obs=torch.zeros(2,273),rnn_states=None))
            self.assertEqual(output['mus'].shape,(2,202))
            self.assertTrue(torch.isfinite(output['mus']).all())
            for key,value in counts.items():
                torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        if not installed:
            self.skipTest('Released modulator weights not installed')

    def test_ppo202_checkpoint_and_optimizer_resume(self):
        torch.set_num_threads(1)
        sys.path.insert(0,str(ROOT/'third_party'))
        from rl_games.common import tr_helpers
        from rl_games.common.algo_observer import DefaultAlgoObserver
        from adadextrack.specialist_agent import SpecialistAgent
        with tempfile.TemporaryDirectory() as folder:
            job=dict(directory=folder,num_envs=2,epochs=2,seed=42,manifest='unused',reference_ids=['0'],latents_dir='unused')
            task,params=configuration(job)
            self.assertEqual(task['env']['initial_object_xy_noise'],.05)
            self.assertEqual(task['env']['initial_object_yaw_noise_degrees'],30)
            self.assertEqual(params['network']['mlp']['units'],[4096,2048,1024,512,256,128])
            params['network']['mlp']['units']=[16,8]
            metadata=dict(schema_version=1,policy_type='modulator',observation_dim=273,action_dim=202,
                          signature='cpu-modulator',model=params['model'],network=copy.deepcopy(params['network']),latents={})
            def make(epochs):
                p=copy.deepcopy(params);cfg=p['config']
                env=ModulatorEnv(FakeEnv(),FakeTracker())
                cfg.update(device='cpu',max_epochs=epochs,mini_epochs=1,save_best_after=0,save_frequency=1,
                           print_stats=False,vec_env=env,features={'observer':DefaultAlgoObserver()},
                           env_info=dict(observation_space=env.observation_space,action_space=env.action_space))
                cfg['reward_shaper']=tr_helpers.DefaultRewardsShaper(**cfg['reward_shaper'])
                return SpecialistAgent(p,metadata,folder)
            agent=make(2)
            try:
                self.assertEqual(agent.train()[1],2)
                tracker=agent.vec_env.tracker
                self.assertEqual(tracker.count.item(),1)
                self.assertTrue(all(p.grad is None for p in tracker.parameters()))
                model=build_model(Path(folder)/'last.pth',action_dim=202)
                with torch.no_grad():
                    pred=model(dict(is_train=False,prev_actions=None,obs=torch.zeros(2,273),rnn_states=None))
                self.assertEqual(pred['mus'].shape,(2,202))
                resumed=make(3)
                try:
                    resumed.restore(Path(folder)/'last.pth')
                    self.assertEqual(resumed.epoch_num,2)
                    self.assertTrue(resumed.optimizer.state_dict()['state'])
                    self.assertEqual(resumed.train()[1],3)
                finally:
                    resumed.writer.close()
            finally:
                agent.writer.close()


if __name__ == '__main__':
    unittest.main()
