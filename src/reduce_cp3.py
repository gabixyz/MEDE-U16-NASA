"""CP3: two-dimensional PCA and autoencoder, with engine-disjoint validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import trustworthiness
from sklearn.neural_network import MLPRegressor

import prepare_cp2 as cp2


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 73)
VIEWS = ("sensors_raw", "operating", "combined_raw",
         "sensors_normalized", "combined_normalized")


def nonoverlapping(frame: pd.DataFrame, width: int) -> pd.DataFrame:
    if width < 2:
        raise ValueError("Window width must be at least two")
    selected = frame.loc[frame.window_end.mod(width).eq(0)].reset_index(drop=True)
    if selected.empty:
        raise ValueError("No complete non-overlapping windows")
    return selected


def feature_views(raw: dict, normalized: dict, regimes: list[str], width: int):
    candidates = {}
    for mode, sources in (("raw", raw), ("normalized", normalized)):
        candidates[mode] = {
            split: cp2.make_windows(frame, width, regimes)
            for split, frame in sources.items()
        }
    scalers = {
        mode: cp2.fit_feature_scaler(frames["fit"])
        for mode, frames in candidates.items()
    }
    scaled = {
        mode: {split: cp2.scale_features(frame, scalers[mode])
               for split, frame in frames.items()}
        for mode, frames in candidates.items()
    }
    result = {}
    for view in VIEWS:
        mode = "normalized" if "normalized" in view else "raw"
        prefix = "S2_" if view == "operating" else "S1_"
        columns = [c for c in scaled[mode]["fit"]
                   if c.startswith(prefix) or
                   (view.startswith("combined") and c.startswith("S2_"))]
        result[view] = {
            split: nonoverlapping(frame[cp2.INDEX + columns], width)
            for split, frame in scaled[mode].items()
        }
    return result, scalers, candidates


def encode_autoencoder(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    if model.activation != "tanh" or len(model.coefs_) != 4:
        raise ValueError("Expected the tanh d-32-2-32-d architecture")
    hidden = np.tanh(x @ model.coefs_[0] + model.intercepts_[0])
    return np.tanh(hidden @ model.coefs_[1] + model.intercepts_[1])


def decode_autoencoder(model: MLPRegressor, z: np.ndarray) -> np.ndarray:
    hidden = np.tanh(z @ model.coefs_[2] + model.intercepts_[2])
    return hidden @ model.coefs_[3] + model.intercepts_[3]


def reconstruction_mse(x, reconstructed, sensor_indexes=None) -> float:
    if x.shape != reconstructed.shape:
        raise ValueError("Reconstruction shape differs from its input")
    if sensor_indexes is not None:
        x, reconstructed = x[:, sensor_indexes], reconstructed[:, sensor_indexes]
    return float(np.mean((x - reconstructed) ** 2))


def neighbor_overlap(left: np.ndarray, right: np.ndarray, k: int = 15) -> float:
    from sklearn.neighbors import NearestNeighbors
    if len(left) != len(right) or len(left) <= k:
        raise ValueError("Matched rows and more than k observations are required")
    neighborhoods = []
    for x in (left, right):
        indexes = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)
        neighborhoods.append([set(row[row != i][:k]) for i, row in enumerate(indexes)])
    return float(np.mean([len(a & b) / k for a, b in zip(*neighborhoods)]))


def evaluation_indexes(n: int, maximum: int = 800) -> np.ndarray:
    if n < 32:
        raise ValueError("At least 32 validation windows are required")
    return np.sort(np.random.default_rng(2026).choice(n, min(n, maximum), replace=False))


def representation_metrics(x, z, reconstructed, indexes, sensor_indexes):
    result = {
        "mse_native": reconstruction_mse(x, reconstructed),
        "trustworthiness_k15": float(trustworthiness(x[indexes], z[indexes], n_neighbors=15)),
    }
    if sensor_indexes:
        result["mse_S1"] = reconstruction_mse(x, reconstructed, sensor_indexes)
    return result


def fit_models(frames: dict, max_iter: int = 300):
    columns = [c for c in frames["fit"] if c not in cp2.INDEX]
    fit = frames["fit"][columns].to_numpy(dtype=float)
    validation = frames["validation"][columns].to_numpy(dtype=float)
    indexes = evaluation_indexes(len(validation))
    sensors = [i for i, c in enumerate(columns) if c.startswith("S1_")]
    pca = PCA(n_components=2, svd_solver="full").fit(fit)
    z_pca = pca.transform(validation)
    reconstructed = pca.inverse_transform(z_pca)
    full_pca = PCA(svd_solver="full").fit(fit)
    result = {
        "features": columns, "dimensions": len(columns),
        "fit_windows": len(fit), "validation_windows": len(validation),
        "evaluation_indexes": indexes.tolist(),
        "pca": representation_metrics(validation, z_pca, reconstructed, indexes, sensors),
        "autoencoder": [],
    }
    result["pca"]["explained_variance_fit_2d"] = float(pca.explained_variance_ratio_.sum())
    result["pca"]["scree_fit"] = full_pca.explained_variance_ratio_.tolist()
    result["pca"]["loadings"] = {
        f"PC{i + 1}": sorted(
            [{"feature": c, "weight": float(v)} for c, v in zip(columns, component)],
            key=lambda record: abs(record["weight"]), reverse=True,
        )[:8]
        for i, component in enumerate(pca.components_)
    }
    embeddings = {"PCA": z_pca}
    for seed in SEEDS:
        model = MLPRegressor(
            hidden_layer_sizes=(32, 2, 32), activation="tanh",
            solver="adam", alpha=0.001, batch_size=256,
            learning_rate_init=0.001, max_iter=max_iter,
            early_stopping=False, tol=1e-5, n_iter_no_change=30,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(fit, fit)
        z = encode_autoencoder(model, validation)
        reconstructed = model.predict(validation)
        if not np.allclose(decode_autoencoder(model, z), reconstructed):
            raise ValueError("Autoencoder bottleneck and predictor disagree")
        metrics = representation_metrics(validation, z, reconstructed, indexes, sensors)
        metrics.update({
            "seed": seed, "iterations": model.n_iter_, "loss_fit": float(model.loss_),
            "warnings": [str(w.message) for w in caught],
            "neighbor_overlap_vs_PCA": neighbor_overlap(z_pca[indexes], z[indexes]),
        })
        for label, mask in (
            ("early_c100", frames["validation"].window_end.le(100).to_numpy()),
            ("later_c100", frames["validation"].window_end.gt(100).to_numpy()),
        ):
            if mask.sum() < 32:
                raise ValueError(f"{label}: insufficient validation windows")
            sub_indexes = evaluation_indexes(int(mask.sum()), 400)
            metrics[label] = representation_metrics(
                validation[mask], z[mask], reconstructed[mask], sub_indexes, sensors
            )
        result["autoencoder"].append(metrics)
        embeddings[f"AE_{seed}"] = z
        print(f"    AE seed {seed}: iter={model.n_iter_}, "
              f"MSE={metrics['mse_native']:.4f}, T15={metrics['trustworthiness_k15']:.4f}",
              flush=True)
    result["stability"] = {
        f"{a}_vs_{b}": neighbor_overlap(embeddings[a][indexes], embeddings[b][indexes])
        for a, b in (("AE_17", "AE_42"), ("AE_17", "AE_73"), ("AE_42", "AE_73"))
    }
    for label, mask in (
        ("early_c100", frames["validation"].window_end.le(100).to_numpy()),
        ("later_c100", frames["validation"].window_end.gt(100).to_numpy()),
    ):
        result["pca"][label] = representation_metrics(
            validation[mask], z_pca[mask], pca.inverse_transform(z_pca[mask]),
            evaluation_indexes(int(mask.sum()), 400), sensors,
        )
    return result, embeddings


def aggregate(records, field):
    values = np.asarray([record[field] for record in records])
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def save_figures(report: dict, embeddings: dict, frames: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for view in VIEWS:
        eigenvalues = np.asarray(report["views"][view]["pca"]["scree_fit"])
        axes[0].plot(np.arange(1, min(15, len(eigenvalues)) + 1),
                     np.cumsum(eigenvalues)[:15], marker=".", label=view)
    axes[0].set(xlabel="Number of PCA components", ylabel="Cumulative fit explained variance",
                ylim=(0, 1.03), title="Scree (fit engines only)")
    axes[0].legend(fontsize=6)
    for view, color in (("combined_raw", "#b36936"), ("combined_normalized", "#207a82")):
        records = report["views"][view]["pca"]["loadings"]["PC1"][:6]
        axes[1].plot([r["weight"] for r in records], marker="o", label=view, color=color)
    axes[1].set(xlabel="Feature rank by absolute loading (different per view)",
                ylabel="PC1 loading", title="Top loadings (details in JSON/CSV)")
    axes[1].legend(fontsize=7)
    fig.savefig(out / "pca_scree_loadings.png", dpi=170)
    plt.close(fig)
    view = "combined_normalized"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    cycles = frames[view]["validation"].window_end.to_numpy()
    for ax, method in zip(axes, ("PCA", "AE_42")):
        z = embeddings[view][method]
        image = ax.scatter(z[:, 0], z[:, 1], c=cycles, cmap="viridis", s=5, alpha=.7)
        ax.set(xlabel="Latent axis 1", ylabel="Latent axis 2",
               title=f"{method}: validation, normalized S1 + S2")
    fig.colorbar(image, ax=axes, label="Observed cycle (not RUL / not a health label)")
    fig.savefig(out / "projections_validation.png", dpi=170)
    plt.close(fig)


def run(raw_dir: Path, out: Path, max_iter: int = 300) -> dict:
    cp2.loader.check_hashes(raw_dir)
    observations = cp2.loader.read_observations(raw_dir / "train_FD002.txt")
    train = pd.DataFrame(
        [[r.engine_id, r.cycle, *r.sensors, *r.settings] for r in observations],
        columns=["engine_id", "cycle"] + cp2.SENSORS + cp2.SETTINGS,
    )
    train.insert(0, "split", "train")
    fit_ids, validation_ids = cp2.engine_partition(train.engine_id)
    raw = {
        "fit": train[train.engine_id.isin(fit_ids)].reset_index(drop=True),
        "validation": train[train.engine_id.isin(validation_ids)].reset_index(drop=True),
    }
    regime_scaler = cp2.fit_regime_scaler(raw["fit"])
    regimes = list(regime_scaler["mapping"].values())
    if len(regimes) != 6:
        raise ValueError("Expected the same six regimes as CP2")
    normalized = {name: cp2.transform_regimes(frame, regime_scaler)
                  for name, frame in raw.items()}
    raw = {name: frame.assign(regime=normalized[name].regime) for name, frame in raw.items()}
    frames, scalers, candidates = feature_views(raw, normalized, regimes, 10)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": "CP3", "dataset": "NASA legacy C-MAPSS FD002",
        "input_sha256": cp2.loader.HASHES, "window": 10,
        "cp2_feature_stride": 1, "model_fit_evaluation_stride": 10,
        "fit_engines": fit_ids, "validation_engines": validation_ids,
        "test_usage": "No test observations or RUL labels loaded; only input hashes checked.",
        "seeds": list(SEEDS), "latent_dimensions": 2,
        "architecture": "d-32-2-32-d; tanh hidden and bottleneck, identity output; Adam",
        "normalization": "Regime z-score and feature scaling fitted on fit engines only.",
        "normalization_caveat": "sensors_normalized uses S2 for preprocessing; not an S1-only baseline.",
        "intervals": "Mean and empirical [min,max] across 3 initialization seeds, NOT confidence intervals.",
        "subperiods": "Validation windows ending at cycle <=100 versus >100; no refitting or future RUL.",
        "hyperparameters": {"max_iter": max_iter, "alpha": .001, "batch_size": 256,
                            "learning_rate_init": .001, "tol": 1e-5, "n_iter_no_change": 30},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "views": {}, "window_sensitivity_fit_only": [], "regime_scaler": regime_scaler,
        "feature_scalers": scalers,
    }
    embeddings = {}
    for view in VIEWS:
        print(f"View: {view}", flush=True)
        result, z = fit_models(frames[view], max_iter=max_iter)
        report["views"][view] = result
        embeddings[view] = z
        pd.DataFrame(result["autoencoder"]).to_json(out / f"seeds_{view}.json", orient="records", indent=2)
        rows = frames[view]["validation"][cp2.INDEX].copy()
        for method, values in z.items():
            rows[f"{method}_1"], rows[f"{method}_2"] = values[:, 0], values[:, 1]
        rows.to_csv(out / f"embedding_{view}.csv", index=False)
        pd.DataFrame([
            {"component": component, **record}
            for component, records in result["pca"]["loadings"].items()
            for record in records
        ]).to_csv(out / f"loadings_{view}.csv", index=False)
    # Window sensitivity is deliberately restricted to fit engines, not model selection on validation.
    for width in (10, 30, 50):
        fit_windows = cp2.make_windows(normalized["fit"], width, regimes)
        fit_scaler = cp2.fit_feature_scaler(fit_windows)
        x = nonoverlapping(cp2.scale_features(fit_windows, fit_scaler), width)
        matrix = x.drop(columns=cp2.INDEX).to_numpy()
        pca = PCA(n_components=2, svd_solver="full").fit(matrix)
        report["window_sensitivity_fit_only"].append({
            "width": width, "overlapping_windows": len(fit_windows),
            "nonoverlapping_windows": len(x), "features": matrix.shape[1],
            "explained_variance_2d_fit": float(pca.explained_variance_ratio_.sum()),
            "mse_fit": reconstruction_mse(matrix, pca.inverse_transform(pca.transform(matrix))),
        })
    report["cp2_overlap_dimensions"] = {
        split: {"rows": len(frame), "features": len(scalers["normalized"]["mean"])}
        for split, frame in candidates["normalized"].items()
    }
    summary = []
    for view, result in report["views"].items():
        ae = result["autoencoder"]
        row = {"view": view, "d": result["dimensions"], "n_fit": result["fit_windows"],
               "n_validation": result["validation_windows"], "pca_evr2": result["pca"]["explained_variance_fit_2d"]}
        for field in ("mse_native", "trustworthiness_k15"):
            row[f"pca_{field}"] = result["pca"][field]
            row.update({f"ae_{field}_{key}": val for key, val in aggregate(ae, field).items()})
        if "mse_S1" in result["pca"]:
            row["pca_mse_S1"] = result["pca"]["mse_S1"]
            row.update({f"ae_mse_S1_{key}": val for key, val in aggregate(ae, "mse_S1").items()})
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "metrics_summary.csv", index=False)
    report["source_code_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("load_fd002.py", "prepare_cp2.py", "reduce_cp3.py")
    }
    (out / "cp3_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    save_figures(report, embeddings, frames, out)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "cp3")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    if args.download:
        cp2.loader.download(args.raw_dir)
    run(args.raw_dir, args.out_dir, args.max_iter)


if __name__ == "__main__":
    main()
