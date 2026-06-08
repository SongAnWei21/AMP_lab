"""G1 29-DOF AMP motion loader — reads CSV files in Isaac Lab articulation order.

CSV format (89 dims, Isaac Lab / Lab order):
    root_pos(3) + root_rot(4) + lin_vel(3) + ang_vel(3)
    + joint_pos(29) + joint_vel(29) + key_body_pos(18)

AMP observation output (74 dims):
    lin_vel(3) + ang_vel(3) + joint_pos(29) + joint_vel(29) + key_body_pos(18)
"""

import numpy as np
import torch


CSV_FPS = 50.0
TIME_BETWEEN_FRAMES = 1.0 / CSV_FPS

CSV_DIM = 89
OFF_LIN_VEL = 7
OFF_ANG_VEL = 10
OFF_JOINT_POS = 13
OFF_JOINT_VEL = 42
OFF_KEY_BODY = 71

N_JOINTS = 29
N_KEY_BODIES = 6


class G1_29DOF_AMPLoader:
    """Loads AMP motion data from CSV files for G1 29-DOF robot.

    CSV is already in Isaac Lab articulation order (GMR→Lab remapping
    was done during PKL→CSV conversion).
    """

    OBS_DIM = 3 + 3 + 29 + 29 + N_KEY_BODIES * 3  # = 74

    def __init__(
        self,
        device,
        motion_files,
        preload_transitions=True,
        num_preload_transitions=1000000,
    ):
        self.device = device

        self.trajectories = []
        self.trajectory_weights = []
        self.trajectory_lens = []
        self.trajectory_num_frames = []
        self.trajectory_frame_durations = []

        for csv_file in motion_files:
            print(f"Loading motion: {csv_file}")
            data = np.loadtxt(csv_file, delimiter=",")
            assert data.shape[1] == CSV_DIM, f"Expected {CSV_DIM} cols, got {data.shape[1]}"

            num_frames = data.shape[0]

            # Extract AMP obs: lin_vel + ang_vel + joint_pos + joint_vel + key_body_pos = 74
            lin_vel = data[:, OFF_LIN_VEL : OFF_ANG_VEL]
            ang_vel = data[:, OFF_ANG_VEL : OFF_JOINT_POS]
            joint_pos = data[:, OFF_JOINT_POS : OFF_JOINT_VEL]
            joint_vel = data[:, OFF_JOINT_VEL : OFF_KEY_BODY]
            key_body_pos = data[:, OFF_KEY_BODY :]

            frames = np.hstack((lin_vel, ang_vel, joint_pos, joint_vel, key_body_pos))
            self.trajectories.append(torch.tensor(frames, dtype=torch.float32, device=device))

            self.trajectory_weights.append(1.0)
            self.trajectory_lens.append((num_frames - 1) * TIME_BETWEEN_FRAMES)
            self.trajectory_num_frames.append(num_frames)
            self.trajectory_frame_durations.append(TIME_BETWEEN_FRAMES)

            print(f"  {num_frames} frames, {self.trajectory_lens[-1]:.1f}s")

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)

        # ---- Preload transitions (space-for-time) ----
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f"Preloading {num_preload_transitions} transitions...")
            traj_idxs = self._weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self._traj_time_sample_batch(traj_idxs)

            self.preloaded_s = self._get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self._get_full_frame_at_time_batch(
                traj_idxs,
                times + TIME_BETWEEN_FRAMES,
            )
            print("  Done preloading.")

    # ---- Sampling helpers ----
    @property
    def observation_dim(self):
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectories)

    def _weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(len(self.trajectories), size=size, p=self.trajectory_weights, replace=True)

    def _traj_time_sample_batch(self, traj_idxs):
        subst = TIME_BETWEEN_FRAMES + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def _slerp(self, frame1, frame2, blend):
        return (1.0 - blend) * frame1 + blend * frame2

    def _get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low = np.floor(p * n).astype(np.int64)
        idx_high = np.ceil(p * n).astype(np.int64)

        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)

        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self._slerp(all_frame_starts, all_frame_ends, blend)

    # ---- Public API ----
    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self._weighted_traj_idx_sample_batch(num_frames)
            times = self._traj_time_sample_batch(traj_idxs)
            return self._get_full_frame_at_time_batch(traj_idxs, times)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generate (s_t, s_{t+1}) pairs for discriminator training."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs]
                s_next = self.preloaded_s_next[idxs]
            else:
                traj_idxs = self._weighted_traj_idx_sample_batch(mini_batch_size)
                times = self._traj_time_sample_batch(traj_idxs)
                s = self._get_full_frame_at_time_batch(traj_idxs, times)
                s_next = self._get_full_frame_at_time_batch(traj_idxs, times + TIME_BETWEEN_FRAMES)
            yield s, s_next

    # ---- Static helpers (for state initialization) ----
    @staticmethod
    def get_joint_pose_batch(frames):
        return frames[:, 6 : 6 + N_JOINTS]

    @staticmethod
    def get_joint_vel_batch(frames):
        return frames[:, 6 + N_JOINTS : 6 + 2 * N_JOINTS]

    @staticmethod
    def get_linear_vel_batch(frames):
        return frames[:, 0:3]

    @staticmethod
    def get_angular_vel_batch(frames):
        return frames[:, 3:6]

    @staticmethod
    def get_key_body_pos_batch(frames):
        return frames[:, 6 + 2 * N_JOINTS :]
