#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_DIRS = [
    Path("dataset/ctr23"),
    Path("dataset/tabarena/reg"),
    Path("dataset/talent_reg"),
]

TARGET_CANDIDATES = ["target", "label", "y", "TARGET", "Label", "Y"]


@dataclass
class ResultRow:
    dataset_dir: str
    dataset_name: str
    n_train: int
    n_test: int
    n_features: int
    r2: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


def infer_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[-1]


def find_csv_files(data_dirs: List[Path]) -> List[Path]:
    csv_files: List[Path] = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        csv_files.extend(sorted(data_dir.glob("*.csv")))
    return csv_files


def evaluate_one_dataset(regressor, csv_path: Path, test_size: float, random_state: int) -> ResultRow:
    try:
        df = pd.read_csv(csv_path)
        target_col = infer_target_column(df)
        df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        t0 = time.time()
        regressor.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        y_pred = regressor.predict(X_test)
        predict_seconds = time.time() - t1

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))

        return ResultRow(
            dataset_dir=csv_path.parent.as_posix(),
            dataset_name=csv_path.name,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            r2=r2,
            rmse=rmse,
            mae=mae,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
        )
    except Exception as exc:
        return ResultRow(
            dataset_dir=csv_path.parent.as_posix(),
            dataset_name=csv_path.name,
            n_train=0,
            n_test=0,
            n_features=0,
            r2=None,
            rmse=None,
            mae=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue,
    worker_out_csv: str,
    model_kwargs: Dict,
    test_size: float,
    random_state: int,
    verbose: bool,
) -> None:
    try:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        from tabicl import TabICLRegressor

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = "cuda:0"
        regressor = TabICLRegressor(**worker_kwargs)

        rows: List[ResultRow] = []
        while True:
            item = task_queue.get()
            if item is None:
                break

            csv_path = Path(item)
            row = evaluate_one_dataset(regressor, csv_path, test_size=test_size, random_state=random_state)
            rows.append(row)

            if verbose:
                print(
                    f"[worker {worker_id} | gpu {gpu_id}] "
                    f"[{row.status}] {row.dataset_name} r2={row.r2}"
                )

        pd.DataFrame([asdict(row) for row in rows]).to_csv(worker_out_csv, index=False)
    except Exception:
        crash_row = pd.DataFrame(
            [
                {
                    "dataset_dir": "__worker__",
                    "dataset_name": f"__WORKER_CRASH__{worker_id}",
                    "n_train": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "r2": None,
                    "rmse": None,
                    "mae": None,
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "status": "fail",
                    "error": traceback.format_exc(),
                }
            ]
        )
        crash_row.to_csv(worker_out_csv, index=False)


def write_summary(summary_path: Path, all_df: pd.DataFrame, csv_files: List[Path], wall_seconds: float) -> None:
    ok_df = all_df[all_df["status"] == "ok"].copy() if len(all_df) else pd.DataFrame()
    failed_df = all_df[all_df["status"] == "fail"].copy() if len(all_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {len(csv_files)}",
        f"processed_datasets: {len(all_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"avg_r2_ok: {ok_df['r2'].mean():.6f}" if len(ok_df) else "avg_r2_ok: (none)",
        f"avg_rmse_ok: {ok_df['rmse'].mean():.6f}" if len(ok_df) else "avg_rmse_ok: (none)",
        f"avg_mae_ok: {ok_df['mae'].mean():.6f}" if len(ok_df) else "avg_mae_ok: (none)",
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        failed_names = ", ".join(failed_df["dataset_name"].astype(str).tolist())
        lines.append(f"failed_datasets: {failed_names}")
    else:
        lines.append("failed_datasets: (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabICLv2 official regressor checkpoint on 3 regression dataset folders with AMD/ROCm multi-GPU.")
    parser.add_argument("--model-path", default="ckpt/TabICLv2/tabicl-regressor-v2-20260212.ckpt")
    parser.add_argument("--out-dir", default="result/TabICLv2_official_regression")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--norm-methods", default="none,power")
    parser.add_argument("--feat-shuffle", default="latin")
    parser.add_argument("--kv-cache", default="kv", choices=["none", "kv", "repr"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = find_csv_files(DATA_DIRS)
    if not csv_files:
        raise FileNotFoundError("No CSV files found under dataset/ctr23, dataset/tabarena/reg, dataset/talent_reg")

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")

    norm_methods = [x.strip() for x in args.norm_methods.split(",") if x.strip()]
    model_kwargs: Dict = {
        "model_path": str(model_path),
        "allow_auto_download": False,
        "batch_size": args.batch_size,
        "n_estimators": args.n_estimators,
        "norm_methods": norm_methods,
        "feat_shuffle_method": args.feat_shuffle,
        "use_amp": True,
        "verbose": False,
        "random_state": args.random_state,
    }
    if args.kv_cache != "none":
        model_kwargs["kv_cache"] = args.kv_cache

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    task_queue: mp.Queue = mp.Queue()
    for csv_path in csv_files:
        task_queue.put(str(csv_path))
    for _ in range(args.workers):
        task_queue.put(None)

    worker_csv_paths: List[Path] = []
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)
        proc = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                task_queue,
                str(worker_csv),
                dict(model_kwargs),
                args.test_size,
                args.random_state,
                args.verbose,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    dfs: List[pd.DataFrame] = []
    for worker_csv in worker_csv_paths:
        if worker_csv.exists():
            dfs.append(pd.read_csv(worker_csv))

    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=ResultRow.__annotations__.keys())
    all_csv = out_dir / "all_regression_results.csv"
    summary_txt = out_dir / "summary.txt"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(summary_txt, all_df, csv_files, wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {summary_txt}")
    print("model_kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
