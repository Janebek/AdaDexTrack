# Third-party notices

The code license at the repository root applies to AdaDexTrack code and does not replace licenses or usage terms of third-party components and resources.

- **IsaacGymEnvs**: tracking environment and supporting utilities were adapted from NVIDIA IsaacGymEnvs. Copyright headers are retained; the upstream BSD-3-Clause license is in `third_party/ISAACGYMENVS_LICENSE.txt`.
- **rl_games**: `third_party/rl_games/` contains the customized network/runtime code used by the released tracker. The upstream MIT license is retained in `third_party/RL_GAMES_LICENSE.txt`. Upstream: https://github.com/Denys88/rl_games.
- **LEAP hand model**: the combined xArm/LEAP model contains modified LEAP hand model files. The available source license is retained in `third_party/LEAP_HAND_LICENSE.txt`. LEAP Hand project: https://leaphand.com/.
- **xArm**: the arm description derives from the xArm robot description; the combined model and modifications are in `assets/xarm_leap/`. Upstream project: https://github.com/xArm-Developer/xarm_ros.
- **Deployment interfaces**: the xArm position-to-velocity example is adapted from [BunnyVisionPro](https://github.com/Dingry/BunnyVisionPro/blob/main/real_control/xarm7_ability.py), and the LEAP joint interface follows the MIT-licensed LEAP deployment wrapper. Copyright and MIT notices are retained in `examples/sim2real/LICENSES.txt`. The external [xArm SDK](https://github.com/xArm-Developer/xArm-Python-SDK), [LEAP Hand API](https://github.com/leap-hand/LEAP_Hand_API), and [FoundationPose](https://github.com/NVlabs/FoundationPose) are installed separately under their own terms.
- **Eye-to-hand calibration**: the example launch integrates the separately installed ROS1 [easy_handeye](https://github.com/IFL-CAMP/easy_handeye), [aruco_ros](https://github.com/pal-robotics/aruco_ros) and [xarm_ros](https://github.com/xArm-Developer/xarm_ros). Their source code is not vendored here; their own licenses apply.
- **Object meshes**: GRAB object meshes and converted files are included in the separately hosted Hugging Face asset bundle. Original terms remain applicable and publication does not establish additional redistribution permission; see https://grab.is.tue.mpg.de/license.html and `docs/resource_distribution.md`.
- **Reference motions**: the local corpus is identified by the authors as DiffH2O-generated motion followed by robot retargeting. References are packaged as a separate Hugging Face resource; generator/checkpoint and dependent-resource terms remain relevant. Source-label provenance remains in the metadata; no blanket data relicensing is asserted.

The proprietary Isaac Gym SDK is not included. Install it separately under its own terms.
