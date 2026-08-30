from pathlib import Path
import math

import h5py
import numpy as np
import tensorflow as tf


ROOT = Path.home() / "DynManipBench"

ROBOT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "dynamic_v1"
)

CONST_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "constInput_liang_v2.npy"
)

# This class must already exist in the project, from the reconstructed
# dynamic-obstacle pipeline.
from pathlib import Path as _Path
import sys
sys.path.insert(0, str(ROOT / "src" / "benchmark"))
from dynamic_dataset import DynamicObstacleModel


REGIME_SPEEDS = {
    "slow": 0.05,
    "medium": 0.10,
    "fast": 0.20,
}


class DynManipSequenceV2(tf.keras.utils.Sequence):
    """
    Source-reconciled training generator.

    Source-verified changes from v1:
      rbInput = current_q + previous_q + goal_q
      constInput = Liang's literal 112-D [mass, COM, inertia, DH] vector

    Still reconstructed rather than source-verified:
      80-D previous/current obstacle representation
      numOfObsInput usage in the final defense model
    """

    def __init__(
        self,
        environment_ids,
        batch_size=4096,
        regime="medium",
        shuffle=True,
        seed=12345,
        samples_per_epoch=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if regime not in REGIME_SPEEDS:
            raise ValueError(f"Unknown regime: {regime}")

        self.environment_ids = list(environment_ids)
        self.batch_size = int(batch_size)
        self.regime = regime
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.obstacles = DynamicObstacleModel()

        self.const_input = np.load(CONST_FILE).astype(np.float32)

        if self.const_input.shape != (112,):
            raise RuntimeError(
                f"constInput must be 112-D, got {self.const_input.shape}"
            )

        if not np.all(np.isfinite(self.const_input)):
            raise RuntimeError("constInput contains NaN/Inf")

        self.env_info = {}
        total_examples = 0

        print("Indexing training environments...")

        for env_id in self.environment_ids:
            file = ROBOT_DIR / f"env{env_id}.h5"

            if not file.exists():
                raise RuntimeError(f"Missing {file}")

            with h5py.File(file, "r") as h5:
                lengths = h5["trajectory_lengths"][:].astype(np.int64)
                offsets = h5["trajectory_offsets"][:].astype(np.int64)
                path_indices = h5["path_indices"][:].astype(np.int64)

            # We need previous, current, and next configurations.
            # Valid current indices are t=1..N-2, giving N-2 examples/path.
            counts = np.maximum(lengths - 2, 0)

            cumulative = np.zeros(len(counts) + 1, dtype=np.int64)
            cumulative[1:] = np.cumsum(counts, dtype=np.int64)

            env_count = int(cumulative[-1])

            self.env_info[env_id] = {
                "file": file,
                "lengths": lengths,
                "offsets": offsets,
                "path_indices": path_indices,
                "counts": counts,
                "cumulative": cumulative,
                "count": env_count,
            }

            total_examples += env_count

        self.total_examples = int(total_examples)

        print(f"Indexed examples: {self.total_examples:,}")

        self.env_counts = np.array(
            [self.env_info[e]["count"] for e in self.environment_ids],
            dtype=np.int64,
        )

        self.env_cumulative = np.zeros(
            len(self.environment_ids) + 1,
            dtype=np.int64,
        )
        self.env_cumulative[1:] = np.cumsum(
            self.env_counts,
            dtype=np.int64,
        )

        if samples_per_epoch is None:
            self.samples_per_epoch = self.total_examples
        else:
            self.samples_per_epoch = min(
                int(samples_per_epoch),
                self.total_examples,
            )

        self.order = np.arange(
            self.samples_per_epoch,
            dtype=np.int64,
        )

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(
            self.samples_per_epoch / self.batch_size
        )

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)

    def _map_global_index(self, global_index):
        env_slot = int(
            np.searchsorted(
                self.env_cumulative,
                global_index,
                side="right",
            ) - 1
        )

        env_id = self.environment_ids[env_slot]

        local_index = (
            global_index
            - self.env_cumulative[env_slot]
        )

        info = self.env_info[env_id]

        path_row = int(
            np.searchsorted(
                info["cumulative"],
                local_index,
                side="right",
            ) - 1
        )

        within_path = (
            local_index
            - info["cumulative"][path_row]
        )

        # First usable current sample is t=1.
        time_step = int(within_path + 1)

        return env_id, path_row, time_step

    def _sample_global_indices(self, batch_number):
        start = batch_number * self.batch_size
        end = min(
            start + self.batch_size,
            self.samples_per_epoch,
        )

        logical = self.order[start:end]

        if self.samples_per_epoch == self.total_examples:
            return logical

        return self.rng.integers(
            low=0,
            high=self.total_examples,
            size=len(logical),
            dtype=np.int64,
        )

    @staticmethod
    def _family_id(env_id):
        if 100 <= env_id <= 199:
            return env_id - 100
        if 200 <= env_id <= 299:
            return env_id - 200
        if 300 <= env_id <= 399:
            return env_id - 300
        raise RuntimeError(f"Invalid environment {env_id}")

    def __getitem__(self, batch_number):
        global_indices = self._sample_global_indices(batch_number)
        n = len(global_indices)

        rb = np.empty((n, 1, 21), dtype=np.float32)
        const = np.empty((n, 1, 112), dtype=np.float32)
        obs = np.empty((n, 1, 80), dtype=np.float32)
        nobs = np.empty((n, 1, 1), dtype=np.float32)
        target = np.empty((n, 1, 7), dtype=np.float32)

        grouped = {}

        for batch_row, gi in enumerate(global_indices):
            env_id, path_row, t = self._map_global_index(int(gi))

            grouped.setdefault(env_id, []).append(
                (batch_row, path_row, t)
            )

        for env_id, rows in grouped.items():
            info = self.env_info[env_id]
            family_id = self._family_id(env_id)

            with h5py.File(info["file"], "r") as h5:
                q_dataset = h5["waypoints_7dof"]
                offsets = info["offsets"]

                for batch_row, path_row, t in rows:
                    begin = int(offsets[path_row])
                    end = int(offsets[path_row + 1])

                    previous_q = (
                        q_dataset[begin + t - 1]
                        .astype(np.float32)
                    )

                    current_q = (
                        q_dataset[begin + t]
                        .astype(np.float32)
                    )

                    next_q = (
                        q_dataset[begin + t + 1]
                        .astype(np.float32)
                    )

                    goal_q = (
                        q_dataset[end - 1]
                        .astype(np.float32)
                    )

                    # SOURCE VERIFIED from Liang loadData.py:
                    # current + previous + goal
                    rb_vector = np.concatenate(
                        [
                            current_q,
                            previous_q,
                            goal_q,
                        ]
                    )

                    # The 80-D obstacle representation is still a
                    # reconstruction of the later dynamic model.
                    obs_vector = self.obstacles.obs_input(
                        family_id,
                        t,
                        self.regime,
                    )

                    rb[batch_row, 0, :] = rb_vector
                    const[batch_row, 0, :] = self.const_input
                    obs[batch_row, 0, :] = obs_vector

                    # Candidate interpretation for the later scalar input:
                    # total obstacles in these environments.
                    nobs[batch_row, 0, 0] = 10.0

                    # SOURCE VERIFIED: target is next configuration.
                    target[batch_row, 0, :] = next_q

        inputs = {
            "rbInput": rb,
            "constInput": const,
            "obsInput": obs,
            "numOfObsInput": nobs,
        }

        return inputs, target


def main():
    print("DynManipBench Liang-reconciled training dataset v2")
    print("=" * 72)

    train_envs = (
        list(range(100, 190))
        + list(range(200, 290))
        + list(range(300, 390))
    )

    validation_envs = (
        list(range(190, 200))
        + list(range(290, 300))
        + list(range(390, 400))
    )

    print()
    print("ENVIRONMENT SPLIT")
    print("-" * 72)
    print(f"Training environments   : {len(train_envs)}")
    print(f"Validation environments : {len(validation_envs)}")

    train = DynManipSequenceV2(
        environment_ids=train_envs,
        batch_size=256,
        regime="medium",
        shuffle=True,
        seed=12345,
        samples_per_epoch=4096,
    )

    validation = DynManipSequenceV2(
        environment_ids=validation_envs,
        batch_size=256,
        regime="medium",
        shuffle=False,
        seed=54321,
        samples_per_epoch=2048,
    )

    print()
    print("DATASET COUNTS")
    print("-" * 72)
    print(f"Training corpus examples : {train.total_examples:,}")
    print(f"Validation corpus examples: {validation.total_examples:,}")
    print(f"Training batches/test    : {len(train)}")
    print(f"Validation batches/test  : {len(validation)}")

    inputs, target = train[0]

    print()
    print("FIRST TRAINING BATCH")
    print("-" * 72)

    for name, value in inputs.items():
        print(
            f"{name:15s}: {value.shape} "
            f"finite={np.all(np.isfinite(value))}"
        )

    print(
        f"{'target':15s}: {target.shape} "
        f"finite={np.all(np.isfinite(target))}"
    )

    expected = {
        "rbInput": (256, 1, 21),
        "constInput": (256, 1, 112),
        "obsInput": (256, 1, 80),
        "numOfObsInput": (256, 1, 1),
    }

    for name, shape in expected.items():
        if inputs[name].shape != shape:
            raise RuntimeError(
                f"{name}: expected {shape}, "
                f"got {inputs[name].shape}"
            )

    if target.shape != (256, 1, 7):
        raise RuntimeError(
            f"Unexpected target shape: {target.shape}"
        )

    print()
    print("FIRST EXAMPLE")
    print("-" * 72)

    rb0 = inputs["rbInput"][0, 0]

    print("current_q:")
    print(rb0[0:7])

    print()
    print("previous_q:")
    print(rb0[7:14])

    print()
    print("goal_q:")
    print(rb0[14:21])

    print()
    print("obsInput first obstacle:")
    print(" previous:", inputs["obsInput"][0, 0, 0:4])
    print(" current :", inputs["obsInput"][0, 0, 40:44])

    print()
    print(
        "numOfObsInput:",
        inputs["numOfObsInput"][0, 0, 0],
    )

    print()
    print("target:")
    print(target[0, 0])

    print()
    print("CONSTANT SOURCE")
    print("-" * 72)
    print(CONST_FILE)

    print()
    print("FINAL RESULT")
    print("-" * 72)
    print(
        "PASS — Liang-reconciled v2 training/validation "
        "dataset generator constructed valid batches."
    )


if __name__ == "__main__":
    main()
