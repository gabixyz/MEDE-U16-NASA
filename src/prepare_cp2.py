from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import load_fd002 as loader


ROOT = Path(__file__).resolve().parents[1]
KEYS = loader.KEYS
SENSORS = loader.SENSORS
SETTINGS = loader.SETTINGS
INDEX = ["split", "engine_id", "window_start", "window_end"]
SEED = 42
WINDOW = 10


def align_views(sensors: pd.DataFrame, operating: pd.DataFrame) -> pd.DataFrame:
    for label, frame, columns in (
        ("S1", sensors, SENSORS), ("S2", operating, SETTINGS)
    ):
        if list(frame.columns) != KEYS + columns:
            raise ValueError(f"{label}: unexpected source schema")
        if frame.empty or frame.isna().any().any():
            raise ValueError(f"{label}: empty source or missing values")
        if not np.isfinite(frame[["engine_id", "cycle"] + columns].to_numpy()).all():
            raise ValueError(f"{label}: non-finite value")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{label}: duplicate join key")
        for column in ("engine_id", "cycle"):
            values = frame[column].to_numpy()
            if (values < 1).any() or (values != np.floor(values)).any():
                raise ValueError(f"{label}: {column} must be a positive integer")
    merged = sensors.merge(
        operating, on=KEYS, how="outer", validate="one_to_one", indicator=True
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("S1/S2: unmatched source keys; no rows may be silently dropped")
    merged = merged.drop(columns="_merge").sort_values(KEYS).reset_index(drop=True)
    for _, group in merged.groupby(["split", "engine_id"], sort=True):
        if not np.array_equal(group["cycle"], np.arange(1, len(group) + 1)):
            raise ValueError("Non-contiguous cycles: interpolation is not enabled")
    return merged


def engine_partition(engine_ids, seed: int = SEED) -> tuple[list[int], list[int]]:
    ids = np.array(sorted(set(engine_ids)), dtype=int)
    if len(ids) < 2:
        raise ValueError("At least two engines are required for an engine-level split")
    shuffled = np.random.default_rng(seed).permutation(ids)
    validation_count = max(1, int(np.ceil(len(ids) * 0.2)))
    return sorted(shuffled[validation_count:].tolist()), sorted(shuffled[:validation_count].tolist())


def regime_keys(frame: pd.DataFrame) -> pd.Series:
    # FD002 has six discrete settings; rounding removes their small numeric jitter.
    return (
        frame["setting_1"].round().astype(int).astype(str)
        + "/"
        + frame["setting_3"].round().astype(int).astype(str)
    )


def fit_regime_scaler(frame: pd.DataFrame) -> dict:
    labelled = frame.assign(regime_key=regime_keys(frame))
    labels = sorted(labelled["regime_key"].unique(), key=lambda key: tuple(map(int, key.split("/"))))
    result = {"mapping": {key: f"R{i}" for i, key in enumerate(labels, 1)}, "parameters": {}}
    for key, group in labelled.groupby("regime_key", sort=True):
        mean = group[SENSORS].mean()
        std = group[SENSORS].std(ddof=0)
        constant = std <= 1e-10 * np.maximum(1, mean.abs())
        result["parameters"][key] = {
            "mean": mean.to_dict(),
            "scale": std.mask(constant, 1.0).to_dict(),
            "constant_sensors": constant[constant].index.tolist(),
            "fit_rows": len(group),
        }
    return result


def transform_regimes(frame: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    keys = regime_keys(frame)
    unknown = sorted(set(keys) - set(scaler["mapping"]))
    if unknown:
        raise ValueError(f"Operating regimes not present in fit engines: {unknown}")
    transformed = frame.copy()
    transformed["regime"] = keys.map(scaler["mapping"])
    for key, indexes in keys.groupby(keys).groups.items():
        parameters = scaler["parameters"][key]
        mean = pd.Series(parameters["mean"])
        scale = pd.Series(parameters["scale"])
        transformed.loc[indexes, SENSORS] = (frame.loc[indexes, SENSORS] - mean) / scale
    return transformed


def feature_schema(regimes: list[str]) -> list[dict]:
    records = []
    for source, columns, statistics in (
        ("S1", SENSORS, ["mean", "std", "slope"]),
        ("S2", SETTINGS, ["mean", "std"]),
    ):
        for column in columns:
            for statistic in statistics:
                unit = "z/cycle" if statistic == "slope" else "z"
                if source == "S2":
                    unit = "NASA setting unit unspecified"
                records.append({
                    "name": f"{source}_{column}_{statistic}",
                    "source": source, "input": column, "statistic": statistic,
                    "pre_scaling_unit": unit, "final_unit": "z",
                })
    records.extend({
        "name": f"S2_share_{regime}", "source": "S2", "input": "operating_regime",
        "statistic": "fraction_of_cycles", "pre_scaling_unit": "fraction", "final_unit": "z",
    } for regime in regimes)
    return records


def make_windows(frame: pd.DataFrame, width: int, regimes: list[str]) -> pd.DataFrame:
    if width < 2:
        raise ValueError("Window width must be at least two cycles")
    columns = [record["name"] for record in feature_schema(regimes)]
    tables = []
    centered_time = np.arange(width, dtype=float) - (width - 1) / 2
    for (split, engine), group in frame.groupby(["split", "engine_id"], sort=True):
        group = group.sort_values("cycle")
        if not np.array_equal(group.cycle, np.arange(1, len(group) + 1)):
            raise ValueError("Windows require contiguous cycles starting at one")
        if len(group) < width:
            continue
        sensor_windows = np.lib.stride_tricks.sliding_window_view(
            group[SENSORS].to_numpy(dtype=float), width, axis=0
        )
        setting_windows = np.lib.stride_tricks.sliding_window_view(
            group[SETTINGS].to_numpy(dtype=float), width, axis=0
        )
        values = {}
        for i, name in enumerate(SENSORS):
            values[f"S1_{name}_mean"] = sensor_windows[:, i, :].mean(axis=1)
            values[f"S1_{name}_std"] = sensor_windows[:, i, :].std(axis=1, ddof=0)
            values[f"S1_{name}_slope"] = (
                sensor_windows[:, i, :] @ centered_time / (centered_time @ centered_time)
            )
        for i, name in enumerate(SETTINGS):
            values[f"S2_{name}_mean"] = setting_windows[:, i, :].mean(axis=1)
            values[f"S2_{name}_std"] = setting_windows[:, i, :].std(axis=1, ddof=0)
        for regime in regimes:
            indicator = group.regime.eq(regime).to_numpy(dtype=float)
            values[f"S2_share_{regime}"] = np.convolve(indicator, np.ones(width) / width, mode="valid")
        table = pd.DataFrame(values, columns=columns)
        table.insert(0, "window_end", group.cycle.to_numpy()[width - 1:])
        table.insert(0, "window_start", group.cycle.to_numpy()[:len(table)])
        table.insert(0, "engine_id", engine)
        table.insert(0, "split", split)
        tables.append(table)
    if not tables:
        raise ValueError(f"No complete windows of length {width}")
    return pd.concat(tables, ignore_index=True)


def fit_feature_scaler(features: pd.DataFrame) -> dict:
    values = features.drop(columns=INDEX)
    mean, std = values.mean(), values.std(ddof=0)
    constant = std <= 1e-10
    return {
        "mean": mean[~constant].to_dict(), "scale": std[~constant].to_dict(),
        "constant_features": constant[constant].index.tolist(),
    }


def scale_features(features: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    columns = list(scaler["mean"])
    result = features[INDEX + columns].copy()
    result[columns] = (
        result[columns] - pd.Series(scaler["mean"])
    ) / pd.Series(scaler["scale"])
    if not np.isfinite(result[columns].to_numpy()).all():
        raise ValueError("Non-finite final features")
    return result


def correlation_block(frame: pd.DataFrame, left: list[str], right: list[str]) -> pd.DataFrame:
    values = frame[left + right].copy()
    # Pearson correlation is undefined for constant columns, including numeric roundoff.
    for column in values:
        if values[column].std(ddof=0) <= 1e-10:
            values[column] = np.nan
    return values.corr().loc[left, right]


def window_comparison(frame: pd.DataFrame, regimes: list[str]) -> list[dict]:
    records = []
    for width in (10, 30, 50):
        windows = make_windows(frame, width, regimes)
        smooth_differences = windows.groupby(["split", "engine_id"])["S1_sensor_02_mean"].diff().abs()
        raw_differences = frame.groupby(["split", "engine_id"])["sensor_02"].diff().abs()
        records.append({
            "width": width, "windows_fit": len(windows),
            "engines_fit": int(windows.engine_id.nunique()),
            "warmup_rows_fit": len(frame) - len(windows),
            "sensor_02_median_abs_step": float(smooth_differences.median()),
            "sensor_02_raw_median_abs_step": float(raw_differences.median()),
        })
    return records


def save_figures(raw: dict, normalized: dict, matrices: dict, report: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "#F7F3EA",
        "axes.facecolor": "#F7F3EA", "savefig.facecolor": "#F7F3EA",
        "text.color": "#17283B", "axes.labelcolor": "#17283B",
    })
    colors = ["#207A82", "#D07836", "#576A91", "#9E4E5B", "#67814A", "#89704D"]
    regimes = report["regimes"]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    selected = ["sensor_02", "sensor_07", "sensor_11", "sensor_15"]
    fit_raw = raw["fit"].assign(regime=normalized["fit"].regime.to_numpy())
    fig = plt.figure(figsize=(14, 7), layout="constrained")
    grid = GridSpec(2, 4, figure=fig, height_ratios=[0.6, 1.7])
    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    fit_count = report["matrices"]["fit"]["rows"]
    validation_count = report["matrices"]["validation"]["rows"]
    test_count = report["matrices"]["test"]["rows"]
    ax.text(0, 0.92, "S1: 21 senzori  +  S2: 3 setari de operare", fontsize=17, weight="bold")
    ax.text(0, 0.58, "Join 1:1 pe (split, motor, ciclu)  ->  z-score pe regim  ->  ferestre cauzale de 10 cicluri", fontsize=12)
    dimensions = report["feature_counts"]["total"]
    ax.text(0, 0.20, f"X_fit: {fit_count:,} x {dimensions}     X_validare: {validation_count:,} x {dimensions}     X_test: {test_count:,} x {dimensions}", fontsize=16, weight="bold")
    ax.text(0.65, 0.92, f"{report['feature_counts']['S1']} trasaturi S1  |  12 trasaturi S2", fontsize=13, color="#207A82")
    ax.text(0.65, 0.70, "RUL si identificatorii nu intra in X.", fontsize=11)
    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[1, 1])
    for axis, frame, title, ylabel in (
        (ax1, fit_raw, "Senzor 02: valori brute", "T24 [degR]"),
        (ax2, normalized["fit"], "Dupa normalizare", "T24 [z, fara unitate]"),
    ):
        boxes = axis.boxplot(
            [frame.loc[frame.regime.eq(regime), "sensor_02"].to_numpy() for regime in regimes],
            tick_labels=regimes, showfliers=False, patch_artist=True, widths=0.65
        )
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.6)
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Regim (motoare fit)")
        axis.grid(axis="y", alpha=0.2)
    for i, (frame, title) in enumerate(((raw["validation"], "Corelatii brute"), (normalized["validation"], "Corelatii dupa z"))):
        axis = fig.add_subplot(grid[1, i + 2])
        block = correlation_block(frame, selected, SETTINGS)
        image = axis.imshow(block.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        axis.set_xticks(range(3), ["Set. 1", "Set. 2", "Set. 3"])
        axis.set_yticks(range(4), [name.replace("sensor_", "S") for name in selected])
        axis.set_title(title, fontsize=14, weight="bold")
        for row in range(4):
            for col in range(3):
                value = block.iloc[row, col]
                label = "0.00" if abs(value) < .005 else f"{value:.2f}"
                axis.text(col, row, label, ha="center", va="center", color="white" if abs(value) > .6 else "#17283B", fontsize=13)
        axis.set_xlabel("Pearson r (validare)")
    fig.colorbar(image, ax=fig.axes[-2:], shrink=0.8, label="r")
    fig.suptitle("Nivelul masuratorilor difera intre regimuri", fontsize=21, weight="bold", x=0.02, ha="left")
    fig.savefig(figures / "cp2_evidence.png", dpi=180)
    plt.close(fig)

    for split in raw:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for axis, frame, title in zip(axes, (raw[split], normalized[split]), ("Brut [degR]", "Normalizat [z]")):
            for regime, color in zip(regimes, colors):
                mask = normalized[split].regime.eq(regime).to_numpy()
                axis.hist(frame.loc[mask, "sensor_02"], bins=35, alpha=0.45, density=True, label=regime, color=color)
            axis.set(title=f"{split}: senzor 02 - {title}", xlabel=title, ylabel="Densitate")
        axes[-1].legend(ncol=3, fontsize=8)
        fig.savefig(figures / f"distributions_{split}.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for axis, frame, title in zip(axes, (raw["fit"], normalized["fit"]), ("Brut [degR]", "Normalizat [z]")):
        for engine, color in zip(report["fit_engines"][:3], colors):
            group = frame[frame.engine_id.eq(engine)]
            axis.plot(group.cycle, group.sensor_02, lw=.7, alpha=.7, color=color, label=f"Motor {engine}")
        axis.set(title=title, xlabel="Ciclu", ylabel="Senzor 02")
        axis.legend()
    fig.suptitle("Exemple de traiectorii - vizualizare, nu dovada validarii unui indicator de sanatate")
    fig.savefig(figures / "trajectories_fit.png", dpi=160)
    plt.close(fig)


def run(raw_dir: Path, out: Path) -> dict:
    source_out = out / "source_views"
    loader_report = loader.run(raw_dir, source_out)
    aligned = {}
    for split in ("train", "test"):
        aligned[split] = align_views(
            pd.read_csv(source_out / f"{split}_sensors.csv"),
            pd.read_csv(source_out / f"{split}_operating.csv"),
        )
    fit_ids, validation_ids = engine_partition(aligned["train"].engine_id)
    raw = {
        "fit": aligned["train"][aligned["train"].engine_id.isin(fit_ids)].reset_index(drop=True),
        "validation": aligned["train"][aligned["train"].engine_id.isin(validation_ids)].reset_index(drop=True),
        "test": aligned["test"].copy(),
    }
    scaler = fit_regime_scaler(raw["fit"])
    regimes = list(scaler["mapping"].values())
    if len(regimes) != 6:
        raise ValueError(f"Expected six FD002 regimes, observed {len(regimes)}")
    normalized = {name: transform_regimes(frame, scaler) for name, frame in raw.items()}
    unscaled = {name: make_windows(frame, WINDOW, regimes) for name, frame in normalized.items()}
    feature_scaler = fit_feature_scaler(unscaled["fit"])
    matrices = {name: scale_features(frame, feature_scaler) for name, frame in unscaled.items()}
    schema = feature_schema(regimes)
    for record in schema:
        record["retained_in_X"] = record["name"] in feature_scaler["mean"]
    pd.DataFrame(schema).to_csv(out / "feature_dictionary.csv", index=False)
    pd.DataFrame({
        "engine_id": fit_ids + validation_ids,
        "partition": ["fit"] * len(fit_ids) + ["validation"] * len(validation_ids),
    }).sort_values("engine_id").to_csv(out / "train_engine_partition.csv", index=False)
    pd.DataFrame(window_comparison(normalized["fit"], regimes)).to_csv(out / "window_comparison_fit.csv", index=False)
    for split in matrices:
        aligned_frame = raw[split].assign(regime=normalized[split].regime)
        aligned_frame.to_csv(out / f"aligned_{split}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        for suffix, features in (("features", unscaled[split]), ("X", matrices[split])):
            features.to_csv(out / f"{suffix}_{split}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
            features.drop(columns=INDEX).describe().T.to_csv(out / f"descriptive_{suffix}_{split}.csv")
        for mode, frame in (("raw", raw[split]), ("normalized", normalized[split])):
            frame[SENSORS + SETTINGS].describe().T.to_csv(out / f"descriptive_{mode}_{split}.csv")
            correlation_block(frame, SENSORS, SETTINGS).to_csv(out / f"correlation_{mode}_{split}.csv")
        correlation_block(
            matrices[split],
            [r["name"] for r in schema if r["source"] == "S1" and r["retained_in_X"]],
            [r["name"] for r in schema if r["source"] == "S2" and r["retained_in_X"]],
        ).to_csv(out / f"correlation_features_{split}.csv")
    report = {
        "checkpoint": "CP2", "dataset": "NASA C-MAPSS FD002", "seed": SEED,
        "sha256": loader_report["sha256"],
        "join_key": KEYS, "join": "outer checked one-to-one; all keys present in both views",
        "interpolation": False, "imputation": False, "resampling": False,
        "time_step": "one observed cycle, physical duration unspecified",
        "window": WINDOW, "stride": 1, "window_is_causal": True,
        "window_choice": "10 cycles: local temporal resolution with smoothing; 30 and 50 compared on fit only. Not a predictive optimum.",
        "fit_engines": fit_ids, "validation_engines": validation_ids,
        "regimes": regimes, "regime_scaler": scaler, "feature_scaler": feature_scaler,
        "feature_counts": {
            "candidates": len(schema),
            "S1": sum(r["source"] == "S1" and r["retained_in_X"] for r in schema),
            "S2": sum(r["source"] == "S2" and r["retained_in_X"] for r in schema),
            "total": len(feature_scaler["mean"]),
        },
        "matrices": {},
        "correlation_example": {},
    }
    for split in matrices:
        frame = raw[split]
        features = matrices[split]
        excluded = sorted(set(frame.engine_id) - set(features.engine_id))
        if excluded:
            raise ValueError(f"{split}: chosen width excludes engines {excluded}")
        raw_corr = correlation_block(frame, ["sensor_02"], ["setting_1"]).iloc[0, 0]
        normalized_corr = correlation_block(normalized[split], ["sensor_02"], ["setting_1"]).iloc[0, 0]
        report["correlation_example"][split] = {"raw": float(raw_corr), "normalized": float(normalized_corr)}
        report["matrices"][split] = {
            "source_rows": len(frame), "rows": len(features),
            "features": len(feature_scaler["mean"]), "engines": int(features.engine_id.nunique()),
            "warmup_rows": len(frame) - len(features), "excluded_engines": excluded,
        }
    (out / "cp2_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    save_figures(raw, normalized, matrices, report, out)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "cp2")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        loader.download(args.raw_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = run(args.raw_dir, args.out_dir)
    for split, matrix in report["matrices"].items():
        print(f"{split}: {matrix['rows']} windows x {matrix['features']} features / {matrix['engines']} engines")
    print(f"Report: {args.out_dir / 'cp2_report.json'}")


if __name__ == "__main__":
    main()
