# Tracking metrics

Each reference is evaluated over exactly **300 frames**. `T_o` and `T_h` are the
time-averaged Euclidean position errors for the object and hand (palm),
respectively. `R_o` and `R_h` are the corresponding time-averaged quaternion
angular errors. Quaternions are normalized before computing
`2 * acos(clamp(abs(dot(q, q_ref)), 0, 1))`, so `q` and `-q` represent the same rotation.

`E_finger` is the mean absolute error across the 16 finger joints at each frame,
averaged over time. These joints occupy indices `6:22` in the environment joint
order. Reference joint positions are reordered using the tracking environment's
inverse permutation before comparison.

The combined errors at each frame are:

```text
position(t) = (object_position_error(t) + hand_position_error(t)) / 2
rotation(t) = (object_rotation_error(t) + hand_rotation_error(t)) / 2
finger(t)   = mean(abs(predicted_finger_qpos(t) - reference_finger_qpos(t)))
```

**Mean-error success** is determined separately for each reference. A reference
succeeds when all three time-averaged combined errors meet their thresholds:
`position <= 0.10 m`, `rotation <= pi/3 rad` (60 degrees), and
`finger <= 2*pi/9 rad` (40 degrees). The overall success rate is the mean of
these per-reference success indicators.

**Completion** applies the same three thresholds at each frame. A frame counts
as completed only when all three conditions hold simultaneously. Completion for
one reference is the number of completed frames divided by 300; overall
Completion is the mean of the per-reference Completion values.

The five tracking errors are also averaged with equal weight per reference.
Offline evaluation additionally reports results by semantic group and object,
using equal reference weights within each group or object. Terminal tables
report position errors in **cm**, angular errors in **rad**, and success and
Completion rates as **percentages**.

Inference summaries and offline evaluation share
`adadextrack/evaluation/metrics.py`, including reference loading, quaternion
normalization, and threshold comparisons. `PaperMetricRecorder` returns a
summary only after receiving all progress steps `1-300`, aligned with reference
frames `0-299`. Missing, duplicated, out-of-order frames or resets during an
episode cause an explicit error. The recorder collects only the five evaluation
errors; it does not change actions, physical states, or training rewards.
Training rewards retain their L1 distances, while evaluation position errors
consistently use L2.
