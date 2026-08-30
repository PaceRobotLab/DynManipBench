from pathlib import Path
import sys, json, csv, datetime
import numpy as np
import tensorflow as tf

ROOT = Path.home() / "DynManipBench"
MODEL_DIR = ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_DIR))

from dynamic_training_dataset_v2 import DynManipSequenceV2
from reconstruct_obstacle_token_model_v1 import build_model

PAIR_FILE = (
    ROOT / "data" / "metadata"
    / "counterfactual_training_clean_v1"
    / "counterfactual_training_examples_clean_v1.npz"
)

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

if set(TRAIN_ENVS) & HELD_OUT_ENVS:
    raise RuntimeError("Training environments overlap held-out environments.")
if len(TRAIN_ENVS) != 270:
    raise RuntimeError(f"Expected 270 training environments, got {len(TRAIN_ENVS)}")

BATCH_SIZE = 256
ORDINARY_SAMPLES_PER_EPOCH = 100_000
PAIR_BATCH = 64
EPOCHS = 100
LR = 1e-4
LAMBDA_PAIR = 1.0
LAMBDA_DELTA = 10.0
SEED = 20260823
CONT = (0, 2, 4, 6)

STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT = ROOT / "runs" / f"mixed_counterfactual_clean_v1_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)


def wrapped_difference(a, b):
    d = b - a
    cols = []
    for j in range(7):
        z = d[:, j]
        if j in CONT:
            z = tf.atan2(tf.sin(z), tf.cos(z))
        cols.append(z)
    return tf.stack(cols, axis=1)


def wrapped_mse(pred, target):
    d = wrapped_difference(pred, target)
    return tf.reduce_mean(tf.square(d))


def make_pair_inputs(d, obs_key, idx):
    return {
        "rbInput": d["rbInput"][idx, None, :].astype(np.float32),
        "constInput": d["constInput"][idx, None, :].astype(np.float32),
        "obsInput": d[obs_key][idx, None, :].astype(np.float32),
        "numOfObsInput": d["numOfObsInput"][idx, None, :].astype(np.float32),
    }


class MixedTrainer:
    def __init__(self, model, ordinary, pairs):
        self.model = model
        self.ordinary = ordinary
        self.pairs = pairs
        self.optimizer = tf.keras.optimizers.Adam(LR)
        self.rng = np.random.default_rng(SEED)

    @tf.function
    def step(self, x, y, xo, xc, yo, yc):
        with tf.GradientTape() as tape:
            # Deterministic passes: Dropout disabled so pair deltas reflect
            # obstacle intervention only; gradients still propagate normally.
            p = tf.reshape(self.model(x, training=False), (-1, 7))
            po = tf.reshape(self.model(xo, training=False), (-1, 7))
            pc = tf.reshape(self.model(xc, training=False), (-1, 7))

            y = tf.reshape(y, (-1, 7))
            yo = tf.reshape(yo, (-1, 7))
            yc = tf.reshape(yc, (-1, 7))

            ordinary_loss = wrapped_mse(p, y)
            orig_pair_loss = wrapped_mse(po, yo)
            cf_pair_loss = wrapped_mse(pc, yc)

            pred_delta = wrapped_difference(po, pc)
            target_delta = wrapped_difference(yo, yc)
            delta_loss = tf.reduce_mean(tf.square(pred_delta - target_delta))

            pair_loss = 0.5 * (orig_pair_loss + cf_pair_loss)
            total = (
                ordinary_loss
                + LAMBDA_PAIR * pair_loss
                + LAMBDA_DELTA * delta_loss
            )

        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            [
                (g, v)
                for g, v in zip(grads, self.model.trainable_variables)
                if g is not None
            ]
        )

        pred_delta_l2 = tf.reduce_mean(tf.norm(pred_delta, axis=1))
        target_delta_l2 = tf.reduce_mean(tf.norm(target_delta, axis=1))

        return (
            total,
            ordinary_loss,
            orig_pair_loss,
            cf_pair_loss,
            delta_loss,
            pred_delta_l2,
            target_delta_l2,
        )

    def train_epoch(self):
        order = self.rng.permutation(len(self.pairs["target_original"]))
        pair_cursor = 0
        vals = []

        for bi in range(len(self.ordinary)):
            x, y = self.ordinary[bi]

            if pair_cursor + PAIR_BATCH > len(order):
                order = self.rng.permutation(len(order))
                pair_cursor = 0

            idx = order[pair_cursor:pair_cursor + PAIR_BATCH]
            pair_cursor += PAIR_BATCH

            xo = make_pair_inputs(self.pairs, "obsInput_original", idx)
            xc = make_pair_inputs(self.pairs, "obsInput_counterfactual", idx)
            yo = self.pairs["target_original"][idx].astype(np.float32)
            yc = self.pairs["target_counterfactual"][idx].astype(np.float32)

            out = self.step(x, y, xo, xc, yo, yc)
            vals.append([float(v) for v in out])

        return np.mean(np.asarray(vals), axis=0)


def paired_eval(model, d):
    idx = np.arange(len(d["target_original"]))
    xo = make_pair_inputs(d, "obsInput_original", idx)
    xc = make_pair_inputs(d, "obsInput_counterfactual", idx)

    po = np.asarray(model(xo, training=False)).reshape(-1, 7)
    pc = np.asarray(model(xc, training=False)).reshape(-1, 7)
    yo = d["target_original"].astype(np.float32)
    yc = d["target_counterfactual"].astype(np.float32)

    dp = pc - po
    dt = yc - yo

    for j in CONT:
        dp[:, j] = np.arctan2(np.sin(dp[:, j]), np.cos(dp[:, j]))
        dt[:, j] = np.arctan2(np.sin(dt[:, j]), np.cos(dt[:, j]))

    pd = np.linalg.norm(dp, axis=1)
    td = np.linalg.norm(dt, axis=1)
    cosine = np.sum(dp * dt, axis=1) / np.maximum(pd * td, 1e-12)

    return {
        "prediction_delta_mean_rad": float(pd.mean()),
        "required_delta_mean_rad": float(td.mean()),
        "delta_ratio_mean": float(np.mean(pd / np.maximum(td, 1e-12))),
        "direction_cosine_mean": float(cosine.mean()),
    }


def main():
    print("DynManipBench CLEAN mixed ordinary + counterfactual training v1")
    print("=" * 80)

    if not PAIR_FILE.exists():
        raise FileNotFoundError(PAIR_FILE)

    pairs = np.load(PAIR_FILE)

    pair_envs = set(int(e) for e in pairs["environment"])
    contamination = sorted(pair_envs & HELD_OUT_ENVS)
    if contamination:
        raise RuntimeError(
            f"Counterfactual training corpus contains held-out environments: "
            f"{contamination}"
        )

    print("Initialization             : RANDOM")
    print("Training environments      : 270")
    print("Held-out environments      : 30 STRICTLY EXCLUDED")
    print("Counterfactual pairs       :", len(pairs["target_original"]))
    print("Pair environments          :", len(pair_envs))
    print("Ordinary samples / epoch   :", ORDINARY_SAMPLES_PER_EPOCH)
    print("lambda_pair                :", LAMBDA_PAIR)
    print("lambda_delta               :", LAMBDA_DELTA)

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
    model = build_model(heads=6, key_dim=149)

    print("Parameters                 :", f"{model.count_params():,}")
    print("BEFORE:", paired_eval(model, pairs))

    trainer = MixedTrainer(model, ordinary, pairs)

    history = []
    best_score = np.inf
    best_epoch = -1

    header = [
        "epoch",
        "loss",
        "ordinary_mse",
        "orig_pair_mse",
        "cf_pair_mse",
        "delta_mse",
        "pred_delta_l2",
        "target_delta_l2",
        "eval_prediction_delta_mean_rad",
        "eval_delta_ratio_mean",
        "eval_direction_cosine_mean",
    ]

    for epoch in range(EPOCHS):
        v = trainer.train_epoch()
        ev = paired_eval(model, pairs)

        row = {
            "epoch": epoch,
            "loss": v[0],
            "ordinary_mse": v[1],
            "orig_pair_mse": v[2],
            "cf_pair_mse": v[3],
            "delta_mse": v[4],
            "pred_delta_l2": v[5],
            "target_delta_l2": v[6],
            "eval_prediction_delta_mean_rad": ev["prediction_delta_mean_rad"],
            "eval_delta_ratio_mean": ev["delta_ratio_mean"],
            "eval_direction_cosine_mean": ev["direction_cosine_mean"],
        }
        history.append(row)

        # Checkpoint criterion uses TRAINING-ONLY quantities.
        score = row["ordinary_mse"] + 10.0 * row["delta_mse"]
        if score < best_score:
            best_score = score
            best_epoch = epoch
            model.save_weights(OUT / "best.weights.h5")

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} "
            f"loss={row['loss']:.6f} "
            f"ordinary={row['ordinary_mse']:.6f} "
            f"delta={row['delta_mse']:.6f} "
            f"pred_delta={ev['prediction_delta_mean_rad']:.6f} "
            f"ratio={ev['delta_ratio_mean']:.4f} "
            f"cos={ev['direction_cosine_mean']:.4f}"
        )

    with (OUT / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(history)

    model.load_weights(OUT / "best.weights.h5")
    final = paired_eval(model, pairs)
    model.save(OUT / "best_model.keras")

    summary = {
        "initialization": "random",
        "parameters": model.count_params(),
        "split": "strict_training_only",
        "ordinary_training_environment_count": len(TRAIN_ENVS),
        "counterfactual_training_pair_count": len(pairs["target_original"]),
        "counterfactual_training_environment_count": len(pair_envs),
        "held_out_environment_ranges_excluded": [
            "190-199",
            "290-299",
            "390-399",
        ],
        "ordinary_samples_per_epoch": ORDINARY_SAMPLES_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "pair_batch": PAIR_BATCH,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "lambda_pair": LAMBDA_PAIR,
        "lambda_delta": LAMBDA_DELTA,
        "seed": SEED,
        "best_epoch_zero_based": best_epoch,
        "training_pair_evaluation": final,
    }

    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("FINAL BEST-CHECKPOINT TRAINING-PAIR EVALUATION")
    print("-" * 80)
    for k, v in final.items():
        print(f"{k:40s}: {v}")
    print("best_epoch_zero_based                   :", best_epoch)
    print("Output:", OUT)
    print()
    print("PASS — clean mixed counterfactual training completed.")


if __name__ == "__main__":
    main()
