"""Checkpoint adapter for the bundled, otherwise unchanged PPO implementation."""

import copy
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from rl_games.algos_torch.a2c_continuous import A2CAgent


class SpecialistAgent(A2CAgent):
    def __init__(self, params, metadata, directory):
        self.metadata = copy.deepcopy(metadata)
        self.directory = Path(directory)
        self._resume_best = None
        super().__init__("a2c", params)

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state.update(adadextrack=copy.deepcopy(self.metadata), learning_rate=self.last_lr,
                     scaler=self.scaler.state_dict(),
                     rng={"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state(),
                          "cuda": torch.cuda.get_rng_state_all() if self.ppo_device.startswith("cuda") else []})
        return state

    def save(self, filename):
        state = self.get_full_state_weights()
        is_best = Path(filename).name == self.config["name"]
        targets = [self.directory / "last.pth"]
        if is_best or (self.epoch_num >= self.max_epochs and not (self.directory / "best.pth").is_file()):
            targets.append(self.directory / "best.pth")
        for target in targets:
            temporary = target.with_suffix(".pth.tmp")
            torch.save(state, str(temporary))
            temporary.replace(target)

    def restore(self, filename, set_epoch=True):
        state = torch.load(str(filename), map_location=self.ppo_device, weights_only=False)
        if state.get("adadextrack", {}).get("signature") != self.metadata["signature"]:
            raise ValueError("Resume checkpoint does not match this group, references or training configuration")
        if state.get("adadextrack", {}).get("latents", self.metadata.get("latents", {})) != self.metadata.get("latents", {}):
            raise ValueError("Resume object latents differ from the checkpoint")
        source_best = Path(filename).resolve().parent / "best.pth"
        target_best = self.directory / "best.pth"
        if source_best.is_file() and not target_best.exists():
            best_state = torch.load(str(source_best), map_location="cpu", weights_only=False)
            if best_state.get("adadextrack", {}).get("signature") != self.metadata["signature"]:
                raise ValueError("The sibling best.pth belongs to a different specialist")
            temporary = target_best.with_suffix(".pth.tmp")
            shutil.copyfile(str(source_best), str(temporary))
            temporary.replace(target_best)
        self.set_full_state_weights(state, set_epoch=set_epoch)
        self.last_lr = float(state["learning_rate"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.last_lr
        self.scaler.load_state_dict(state.get("scaler", {}))
        self._resume_best = self.last_mean_rewards
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] and self.ppo_device.startswith("cuda"):
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])

    def update_epoch(self):
        # The vendored train() initializes its reward meters on every call.
        # Preserve the checkpoint's historical best across that initialization.
        if self._resume_best is not None:
            self.last_mean_rewards = self._resume_best
            self._resume_best = None
        return super().update_epoch()
