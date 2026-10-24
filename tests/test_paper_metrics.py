"""Regression coverage for the shared terminal/offline evaluation protocol."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from adadextrack.evaluation.metrics import (
    LINKS, PaperMetricRecorder, compute_paper_metrics, aggregate_metrics, load_reference,
)


class PaperMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pose = np.zeros((300, 7))
        self.pose[:, 6] = 1
        self.qpos = np.zeros((300, 22))
        self.paths = []
        for index in range(2):
            path = Path(self.directory.name)/('reference_'+str(index)+'.npy')
            np.save(path, dict(object_transl=self.pose[:,:3], object_rot_quat=self.pose[:,3:],
                robot_delta_states_weights_np=self.qpos,
                link_key_to_link_pos={key:self.pose[:,:3] for key in LINKS},
                link_key_to_link_rot={'palm_lower':self.pose[:,3:]}))
            self.paths.append(path)

    def test_l2_and_last_frame_are_used_online_and_offline(self):
        obj = np.repeat(self.pose[:,None,:],2,axis=1)
        hand = obj.copy()
        hand[:,:,:2] = (.03,.04)
        joints = np.repeat(self.qpos[:,None,:],2,axis=1)
        obj[-1,:,0] = .30
        # Equivalent quaternion signs must produce the same rotation error.
        hand[:,:,3:] *= -1
        recorder = PaperMetricRecorder(['0','1'],self.paths)
        for frame in range(299):
            self.assertIsNone(recorder.append(np.full(2,frame+1),obj[frame],hand[frame],joints[frame]))
        self.assertIsNone(recorder.rows)
        online = recorder.append(np.full(2,300),obj[-1],hand[-1],joints[-1])
        rows = [dict(reference_id=str(index),**compute_paper_metrics(
            obj[:,index],hand[:,index],joints[:,index],load_reference(path)))
            for index,path in enumerate(self.paths)]
        self.assertEqual(online,aggregate_metrics(rows,['0','1']))
        self.assertEqual(recorder.rows,rows)
        self.assertAlmostEqual(online['wrist_pos_error_m'],.05)
        self.assertEqual(online['wrist_rot_error_rad'],0)
        self.assertEqual(rows[0]['completed_frames'],299)
        self.assertEqual(online['completion_rate'],299/300)

    def test_skipped_frames_and_asynchronous_resets_rejected(self):
        recorder = PaperMetricRecorder(['0','1'],self.paths)
        pose = np.repeat(self.pose[:1],2,axis=0)
        joints = self.qpos[:2]
        for progress in ([0,0],[2,2],[1,2]):
            with self.assertRaises(ValueError):
                recorder.append(progress,pose,pose,joints)
        self.assertIsNone(recorder.append([1,1],pose,pose,joints))
        with self.assertRaises(ValueError):
            recorder.append([1,1],pose,pose,joints)

    def test_duplicate_references_rejected(self):
        with self.assertRaises(ValueError):
            PaperMetricRecorder(['0','0'],self.paths)



if __name__=='__main__':
    unittest.main()
