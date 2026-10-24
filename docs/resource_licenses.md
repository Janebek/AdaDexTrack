# Resource sources and terms

This file records sources and retained terms. It does not supply missing
redistribution permission or grant a new license over third-party resources.
The Hugging Face `license: other` metadata describes this component-specific
arrangement; it is not an alternative to the original licenses.

| Component | Source | Terms and attribution |
|---|---|---|
| GRAB object meshes and their converted visual/collision files | https://grab.is.tue.mpg.de/ | Original terms apply: https://grab.is.tue.mpg.de/license.html |
| xArm model and combined xArm/LEAP description | https://github.com/xArm-Developer/xarm_ros | Retain source notices; no new license over upstream material is granted here |
| LEAP hand model | https://leaphand.com/ | Available source notice copied to `notices/LEAP_HAND_LICENSE.txt` |
| IsaacGymEnvs-derived assets/utilities | https://github.com/isaac-sim/IsaacGymEnvs | Retained NVIDIA BSD notice in `notices/ISAACGYMENVS_LICENSE.txt`; the SDK itself is not included |
| Object latents | Mesh-conditioned representations used by AdaDexTrack | Mesh and encoder provenance remain relevant; no blanket third-party relicensing is asserted |
| Generated robot references | DiffH2O generation followed by robot retargeting | Generator source: https://github.com/facebookresearch/diffh2o; its code and dependent datasets retain their respective terms |
| Recovered AE input meshes | Original GRAB contact-mesh directory, 57-file snapshot | The same source mesh terms apply; this snapshot is not a new grant of rights |
| Object AE checkpoint | AdaDexTrack object autoencoder training | Provided separately with original training state; input-mesh terms remain relevant |
| Modulator checkpoints | AdaDexTrack PPO training with frozen BC trackers | Original trained weights are provided separately; no new license over third-party training resources is asserted |
| Example specialist checkpoints | AdaDexTrack semantic-group PPO training | Original trained weights and training state for the seven example references; no blanket third-party relicensing is asserted |
| BC tracker checkpoint | AdaDexTrack training | Trained weights are provided as a separate artifact; the code license does not itself determine the terms of all training resources |

GRAB's published terms explicitly include 3D meshes and prohibit making its
Data & Software available to third parties without prior written permission;
they also contain a No Distribution provision. These restrictions remain
applicable. Public availability of an archive does not replace that permission.

GRAB cites ContactDB for object resources. ContactDB's original-model provenance
sheet lists different terms for individual original models:
https://docs.google.com/spreadsheets/d/1v1BqZJOKQ0chPYkBFVJvfToLU7KWs-2RRb2tsJ5IDP8/edit.
An original-model license must not be assumed to cover a processed model from a
different distribution without verifying its provenance and terms.

References in this package are identified by the authors as generated motions,
not a direct release of the original GRAB human-motion dataset. The dictionaries
contain robot/object transforms rather than mesh vertices. This distinction
alone does not establish a particular license for generated outputs. Sampling
conditions and dependent model/data terms have not been audited per generation
run in this release-preparation process.

The source and rights review is not a statement that every file has received
separate redistribution permission. No blanket CC BY-NC or BSD license is
assigned to the entire bundle. Questions about additional permissions should be
resolved with the relevant rightsholders.
