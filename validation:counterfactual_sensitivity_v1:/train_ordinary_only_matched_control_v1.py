from pathlib import Path
import sys
import csv
import json
import datetime

import numpy as np
import tensorflow as tf

ROOT = Path.home() / "DynManipBench"
MODEL_DIR = ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from dynamic_training_dataset_v2 import DynManipSequenceV2
from reconstruct_obstacle_token_model_v1 import build_model

TRAIN_ENVS = (
    list(range(100, 190))
    + list(range(200, 290))
    + list(range(300, 390))
)

HELD_OUT_ENVS = (
    set(range(190, 200))
    | set(range(290, 300))
    | set(range(390, 400))
)

BATCH_SIZE = 256
ORDINARY_SAMPLES_PER_EPOCH = 100_000
EPOCHS = 100
LR = 1e-4
SEED = 20260823
CONTINUOUS_JOINTS = (0, 2, 4, 6)

STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT = ROOT / "runs" / f"ordinary_only_matched_control_v1_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)


def wrapped_difference(pred, target):
    d = pred - target
    cols = []

    for j in range(7):
        z = d[:, j]
        if j in CONTINUOUS_JOINTS:
            z = tf.atan2(tf.sin(z), tf.cos(z))
        cols.append(z)

    return tf.stack(cols, axis=1)


def wrapped_mse(pred, target):
    d = wrapped_difference(pred, target)
    return tf.reduce_mean(tf.square(d))


class OrdinaryTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(LR)

    @tf.function
    def step(self, x, y):
        with tf.GradientTape() as tape:
            pred = tf.reshape(
                self.model(x, training=False),
                (-1, 7),
            )
            target = tf.reshape(y, (-1, 7))
            loss = wrapped_mse(pred, target)

        grads = tape.gradient(
            loss,
            self.model.trainable_variables,
        )

        self.optimizer.apply_gradients(
            [
                (g, v)
                for g, v in zip(
                    grads,
                    self.model.trainable_variables,
                )
                if g is not None
            ]
        )

        return loss

    def train_epoch(self, dataset):
        losses = []

        for batch_index in range(len(dataset)):
            x, y = dataset[batch_index]
            loss = self.step(x, y)
            losses.append(float(loss))

        return float(np.mean(losses))


def main():
    print(
        "DynManipBench MATCHED ordinary-only "
        "control training v1"
    )
    print("=" * 80)

    overlap = set(TRAIN_ENVS) & HELD_OUT_ENVS

    if overlap:
        raise RuntimeError(
            "Training environments overlap held-out "
            f"environments: {sorted(overlap)}"
        )

    if len(TRAIN_ENVS) != 270:
        raise RuntimeError(
            f"Expected 270 training environments, "
            f"got {len(TRAIN_ENVS)}"
        )

    print("Initialization             : RANDOM")
    print("Training environments      : 270")
    print("Held-out environments      : 30 STRICTLY EXCLUDED")
    print(
        "Ordinary samples / epoch   :",
        ORDINARY_SAMPLES_PER_EPOCH,
    )
    print("Counterfactual pairs       : 0")
    print("Epochs                     :", EPOCHS)
    print("Learning rate              :", LR)
    print("Seed                       :", SEED)
    print()

    ordinary = DynManipSequenceV2(
        environment_ids=TRAIN_ENVS,
        batch_size=BATCH_SIZE,
        regime="medium",
        shuffle=True,
        seed=SEED,
        samples_per_epoch=ORDINARY_SAMPLES_PER_EPOCH,
    )

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = build_model(
        heads=6,
        key_dim=149,
    )

    print("Parameters                 :", f"{model.count_params():,}")
    print()

    trainer = OrdinaryTrainer(model)

    history = []
    best_loss = np.inf
    best_epoch = -1

    for epoch in range(EPOCHS):
        loss = trainer.train_epoch(ordinary)

        row = {
            "epoch": epoch,
            "ordinary_mse": loss,
        }
        history.append(row)

        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            model.save_weights(
                OUT / "best.weights.h5"
            )

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} "
            f"ordinary={loss:.6f}"
        )

    with (OUT / "history.csv").open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "ordinary_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    model.load_weights(
        OUT / "best.weights.h5"
    )

    model.save(
        OUT / "best_model.keras"
    )

    summary = {
        "experiment":
            "matched_ordinary_only_control",
        "initialization":
            "random",
        "parameters":
            int(model.count_params()),
        "split":
            "strict_training_only",
        "training_environment_count":
            len(TRAIN_ENVS),
        "training_environment_ranges": [
            "100-189",
            "200-289",
            "300-389",
        ],
        "held_out_environment_ranges_excluded": [
            "190-199",
            "290-299",
            "390-399",
        ],
        "ordinary_samples_per_epoch":
            ORDINARY_SAMPLES_PER_EPOCH,
        "batch_size":
            BATCH_SIZE,
        "counterfactual_pairs":
            0,
        "epochs":
            EPOCHS,
        "learning_rate":
            LR,
        "seed":
            SEED,
        "best_epoch_zero_based":
            best_epoch,
        "best_training_ordinary_mse":
            float(best_loss),
        "comparison_target":
            "mixed_counterfactual_clean_v1_20260824-124312",
        "note": (
            "Matched control for the clean mixed model. "
            "Uses the same architecture, random initialization, "
            "270 training environments, wrapped ordinary loss, "
            "ordinary sampling budget, optimizer, learning rate, "
            "epoch count, and seed, but no counterfactual pairs "
            "or counterfactual/delta losses."
        ),
    }

    (OUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print("FINAL MATCHED CONTROL SUMMARY")
    print("-" * 80)
    print(
        "best_epoch_zero_based        :",
        best_epoch,
    )
    print(
        "best_training_ordinary_mse   :",
        best_loss,
    )
    print("Output                      :", OUT)
    print()
    print(
        "PASS — matched ordinary-only "
        "control training completed."
    )


if __name__ == "__main__":
    main()
