"""Train the symbol classifier.

  python -m model.train                       # Keras CNN (needs TensorFlow)
  python -m model.train --backend sklearn     # MLP on HOG features, no TensorFlow (fallback)
  python -m model.train --external           # also use data/external/*.npz
  python -m model.train --epochs 40 --real-repeat 8

Data: data/synthetic/train.npz (make_synthetic), real camera crops from
data/crops_real/ (crop_sheet), and optionally data/external/*.npz
(import_datasets). Real crops are repeated --real-repeat times because they
are few but match the camera exactly; 20 % of the photos they came from are
held out to measure accuracy on real data (gate G2: >= 95 %).

Writes models/<name>.keras (or .joblib) and models/<name>.json (class list,
settings and scores).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from model.data import holdout_mask, load_npz, load_real_crops
from pipeline.symbols import load_symbols


def build_keras(num_classes: int, size: int):
    """Small CNN, about 149,000 weights for 159 classes.

    conv32-conv32-pool | conv64-conv64-pool | 1x1 conv32-pool | dropout | dense.
    The last pooling keeps a 4x4 grid, so where a stroke sits (top or bottom
    of the crop) still reaches the dense layer.
    """
    import tensorflow as tf
    keras = tf.keras
    layers = keras.layers

    def block(x, filters, k=3):
        x = layers.Conv2D(filters, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    inp = keras.Input((size, size, 1))
    x = layers.RandomTranslation(0.06, 0.06, fill_mode="constant")(inp)   # training only
    x = layers.RandomZoom(0.08, fill_mode="constant")(x)                  # training only
    x = block(x, 32)
    x = block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = block(x, 64)
    x = block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = block(x, 32, k=1)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def keras_param_count(num_classes: int) -> int:
    """Weights of build_keras without importing TensorFlow (for docs and checks)."""
    convs = 3 * 3 * 1 * 32 + 3 * 3 * 32 * 32 + 3 * 3 * 32 * 64 + 3 * 3 * 64 * 64 + 1 * 1 * 64 * 32
    bn = 4 * (32 + 32 + 64 + 64 + 32)
    dense = 4 * 4 * 32 * num_classes + num_classes
    return convs + bn + dense


def per_class_report(y_true, y_pred, classes, top: int = 10) -> str:
    lines = []
    errs = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            errs[(classes[t], classes[p])] = errs.get((classes[t], classes[p]), 0) + 1
    for (t, p), n in sorted(errs.items(), key=lambda kv: -kv[1])[:top]:
        lines.append(f"  {t:>14s} read as {p:<14s} x{n}")
    return "\n".join(lines) if lines else "  (no mistakes)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("keras", "sklearn"), default="keras")
    ap.add_argument("--name", default=config.MODEL_NAME)
    ap.add_argument("--synthetic", type=Path, nargs="*", default=[Path(config.SYNTH_DIR) / "train.npz"])
    ap.add_argument("--real", type=Path, default=Path(config.CROPS_REAL_DIR))
    ap.add_argument("--real-repeat", type=int, default=5)
    ap.add_argument("--external", action="store_true", help="add data/external/*.npz")
    ap.add_argument("--external-share", type=float, default=0.25,
                    help="at most this share of the training set comes from external data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    table = load_symbols()
    classes = table.classes
    C = len(classes)

    Xs, ys = [], []
    for p in args.synthetic:
        if p.exists():
            X, y = load_npz(p, table)
            Xs.append(X)
            ys.append(y)
            print(f"synthetic {p}: {len(X)} crops")
        else:
            print(f"missing {p}: run python -m model.make_synthetic first")
    if not Xs:
        return 1
    Xsyn, ysyn = np.concatenate(Xs), np.concatenate(ys)
    val_syn = rng.random(len(Xsyn)) < 0.05

    Xr, yr, groups = load_real_crops(args.real, table)
    hold = holdout_mask(groups) if len(Xr) else np.zeros(0, bool)
    print(f"real crops: {len(Xr)} ({hold.sum()} held out for testing)")

    parts_X = [Xsyn[~val_syn]] + [Xr[~hold]] * args.real_repeat
    parts_y = [ysyn[~val_syn]] + [yr[~hold]] * args.real_repeat
    n_ext = 0
    if args.external:
        limit = int(args.external_share * sum(len(x) for x in parts_X))
        ext = sorted(Path(config.EXTERNAL_DIR).glob("*.npz"))
        for p in ext:
            X, y = load_npz(p, table)
            take = rng.permutation(len(X))[:max(0, limit - n_ext)]
            parts_X.append(X[take])
            parts_y.append(y[take])
            n_ext += len(take)
            print(f"external {p.name}: using {len(take)} of {len(X)}")
    Xtr = np.concatenate(parts_X).astype(np.float32) / 255.0
    ytr = np.concatenate(parts_y)
    order = rng.permutation(len(Xtr))
    Xtr, ytr = Xtr[order], ytr[order]
    Xv_syn = Xsyn[val_syn].astype(np.float32) / 255.0
    yv_syn = ysyn[val_syn]
    Xv_real = Xr[hold].astype(np.float32) / 255.0
    yv_real = yr[hold]
    missing = sorted(set(range(C)) - set(ytr.tolist()))
    if missing:
        print(f"warning: no training crops for {[classes[i] for i in missing]}")
    print(f"training on {len(Xtr)} crops, {C} classes, backend {args.backend}")

    models_dir = Path(config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.backend == "keras":
        import tensorflow as tf
        keras = tf.keras
        keras.utils.set_random_seed(args.seed)
        model = build_keras(C, config.CROP_SIZE)
        print(f"model: {model.count_params()} weights")
        if len(Xv_real):
            val = (Xv_real[..., None], yv_real)
        else:
            val = (Xv_syn[..., None], yv_syn)
        cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
        model.fit(Xtr[..., None], ytr, validation_data=val, epochs=args.epochs,
                  batch_size=args.batch, callbacks=cb, verbose=2)
        predict = lambda X: np.argmax(model.predict(X[..., None], verbose=0, batch_size=512), axis=1)
        model.save(models_dir / f"{args.name}.keras")
        weights = int(model.count_params())
    else:
        import joblib
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4, batch_size=256,
                              learning_rate_init=1e-3, max_iter=max(10, args.epochs * 2),
                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
                              random_state=args.seed, verbose=False)
        from pipeline.classify import sklearn_features
        model.fit(sklearn_features(Xtr), ytr)
        predict = lambda X: model.classes_[np.argmax(model.predict_proba(sklearn_features(X)), axis=1)]
        joblib.dump(model, models_dir / f"{args.name}.joblib", compress=3)
        weights = int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))
    minutes = (time.time() - t0) / 60

    scores = {}
    if len(Xv_syn):
        p = predict(Xv_syn)
        scores["synthetic_val_accuracy"] = float((p == yv_syn).mean())
    if len(Xv_real):
        p = predict(Xv_real)
        scores["real_val_accuracy"] = float((p == yv_real).mean())
        print("most common mistakes on held-out real crops:")
        print(per_class_report(yv_real, p, classes))
    info = {
        "name": args.name, "backend": args.backend, "classes": classes,
        "crop_size": config.CROP_SIZE, "window_scale": config.WINDOW_SCALE,
        "features": "hog+pix" if args.backend == "sklearn" else "pixels",
        "weights": weights, "trained": time.strftime("%Y-%m-%d %H:%M"),
        "minutes": round(minutes, 1),
        "data": {"synthetic": int((~val_syn).sum()), "real": int((~hold).sum()),
                 "real_repeat": args.real_repeat, "external": n_ext},
        **scores,
    }
    (models_dir / f"{args.name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "classes"}, indent=1))
    if "real_val_accuracy" in scores and scores["real_val_accuracy"] < 0.95:
        print("real-crop accuracy is under the 95 % gate: photograph more sheets or check the confusions above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
