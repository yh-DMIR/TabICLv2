#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


DEFAULT_BENCHMARKS = ["openml_cc18_csv", "TabFSBeach", "tabzilla_csv", "talent_csv"]
TARGET_CANDIDATES = ["target", "label", "class", "y", "TARGET", "Label", "Class", "Y"]


@dataclass
class ResultRow:
    benchmark: str
    dataset_id: str
    n_train: int
    n_test: int
    n_features: int
    n_classes: Optional[int]
    accuracy: Optional[float]
    f1_weighted: Optional[float]
    logloss: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


def sanitize_dataset_id(train_path: Path) -> str:
    match = re.search(r"(OpenML-ID-\d+)", str(train_path))
    return match.group(1) if match else train_path.parent.name


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in train_df.columns:
            return col
    extra = [col for col in train_df.columns if col not in test_df.columns]
    if len(extra) == 1:
        return extra[0]
    return train_df.columns[-1]


def pair_size(train_csv: Path, test_csv: Path) -> int:
    try:
        return train_csv.stat().st_size + test_csv.stat().st_size
    except OSError:
        return 0


def find_dataset_pairs(benchmark_dir: Path, largest_first: bool) -> List[Tuple[Path, Path, int]]:
    pairs: List[Tuple[Path, Path, int]] = []
    for train_csv in benchmark_dir.rglob("*_train.csv"):
        test_csv = train_csv.with_name(train_csv.name.replace("_train.csv", "_test.csv"))
        if test_csv.exists():
            pairs.append((train_csv, test_csv, pair_size(train_csv, test_csv)))
    if largest_first:
        return sorted(pairs, key=lambda x: (-x[2], str(x[0])))
    return sorted(pairs, key=lambda x: str(x[0]))


def build_tasks(root: Path, benchmark_names: List[str], largest_first: bool) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    per_benchmark: Dict[str, List[Tuple[Path, Path, int]]] = {}
    discovered: Dict[str, int] = {}

    for benchmark in benchmark_names:
        benchmark_dir = root / benchmark
        pairs = find_dataset_pairs(benchmark_dir, largest_first) if benchmark_dir.exists() else []
        per_benchmark[benchmark] = pairs
        discovered[benchmark] = len(pairs)

    tasks: List[Tuple[str, str, str]] = []
    while True:
        batch: List[Tuple[str, Path, Path, int]] = []
        for benchmark in benchmark_names:
            if per_benchmark[benchmark]:
                train_csv, test_csv, size_bytes = per_benchmark[benchmark].pop(0)
                batch.append((benchmark, train_csv, test_csv, size_bytes))
        if not batch:
            break
        if largest_first:
            batch.sort(key=lambda x: (-x[3], x[0], str(x[1])))
        for benchmark, train_csv, test_csv, _ in batch:
            tasks.append((benchmark, str(train_csv), str(test_csv)))
    return tasks, discovered


def evaluate_one_dataset(clf, benchmark: str, train_csv: Path, test_csv: Path) -> ResultRow:
    dataset_id = sanitize_dataset_id(train_csv)
    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        target_col = infer_target_column(train_df, test_df)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        if target_col in test_df.columns:
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
        else:
            X_test = test_df
            y_test = None

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        if y_test is not None:
            try:
                proba = clf.predict_proba(X_test)
                y_pred = clf.classes_[np.argmax(proba, axis=1)]
                ll = log_loss(y_test, proba, labels=clf.classes_)
            except Exception:
                y_pred = clf.predict(X_test)
                ll = None
        else:
            y_pred = clf.predict(X_test)
            ll = None
        predict_seconds = time.time() - t1

        if y_test is not None:
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
        else:
            accuracy = None
            f1_weighted = None

        return ResultRow(
            benchmark=benchmark,
            dataset_id=dataset_id,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            n_classes=int(y_train.nunique()),
            accuracy=float(accuracy) if accuracy is not None else None,
            f1_weighted=float(f1_weighted) if f1_weighted is not None else None,
            logloss=float(ll) if ll is not None else None,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
        )
    except Exception as exc:
        return ResultRow(
            benchmark=benchmark,
            dataset_id=dataset_id,
            n_train=0,
            n_test=0,
            n_features=0,
            n_classes=None,
            accuracy=None,
            f1_weighted=None,
            logloss=None,
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
    verbose: bool,
) -> None:
    try:
        gpu_id_str = str(gpu_id)
        os.environ["HIP_VISIBLE_DEVICES"] = gpu_id_str
        os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPU backend is not available in this worker.")

        from tabicl import TabICLClassifier

        worker_kwargs = dict(model_kwargs)
        worker_kwargs["device"] = "cuda:0"
        clf = TabICLClassifier(**worker_kwargs)

        rows: List[ResultRow] = []
        while True:
            item = task_queue.get()
            if item is None:
                break

            benchmark, train_csv, test_csv = item
            row = evaluate_one_dataset(clf, benchmark, Path(train_csv), Path(test_csv))
            rows.append(row)

            if verbose:
                if row.status == "ok":
                    print(f"[worker {worker_id} | gpu {gpu_id}] [ok] {benchmark}/{row.dataset_id} acc={row.accuracy}")
                else:
                    print(f"[worker {worker_id} | gpu {gpu_id}] [fail] {benchmark}/{row.dataset_id} error={row.error}")

        columns = list(ResultRow.__annotations__.keys())
        worker_df = pd.DataFrame([asdict(row) for row in rows]) if rows else pd.DataFrame(columns=columns)
        worker_df.to_csv(worker_out_csv, index=False)
    except Exception:
        pd.DataFrame(
            [
                {
                    "benchmark": "__worker__",
                    "dataset_id": f"__WORKER_CRASH__{worker_id}",
                    "n_train": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "n_classes": None,
                    "accuracy": None,
                    "f1_weighted": None,
                    "logloss": None,
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "status": "fail",
                    "error": traceback.format_exc(),
                }
            ]
        ).to_csv(worker_out_csv, index=False)


def write_summary(summary_path: Path, result_df: pd.DataFrame, discovered_pairs: int, wall_seconds: float) -> None:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_pairs: {discovered_pairs}",
        f"processed_pairs: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        f"avg_accuracy_ok: {ok_df['accuracy'].dropna().mean():.6f}" if len(ok_df) and ok_df["accuracy"].notna().any() else "avg_accuracy_ok: (none)",
        f"avg_f1_weighted_ok: {ok_df['f1_weighted'].dropna().mean():.6f}" if len(ok_df) and ok_df["f1_weighted"].notna().any() else "avg_f1_weighted_ok: (none)",
        f"avg_logloss_ok: {ok_df['logloss'].dropna().mean():.6f}" if len(ok_df) and ok_df["logloss"].notna().any() else "avg_logloss_ok: (none)",
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        failed_names = ", ".join(sorted(set(failed_df["dataset_id"].dropna().astype(str).tolist())))
        lines.append(f"failed_datasets: {failed_names}")
    else:
        lines.append("failed_datasets: (none)")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="limix")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--out-dir", default="result/TabICLv2_classification_amd")
    parser.add_argument("--model-path", default="ckpt/TabICLv2/tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--norm-methods", default="none,power")
    parser.add_argument("--feat-shuffle", default="latin")
    parser.add_argument("--kv-cache", default="kv", choices=["none", "kv", "repr"])
    parser.add_argument("--softmax-temp", type=float, default=0.9)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-class-shift", action="store_true")
    parser.add_argument("--no-average-logits", action="store_true")
    parser.add_argument("--small-first", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    model_path = Path(args.model_path).expanduser()
    try:
        model_path = model_path.resolve()
    except Exception:
        pass
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_names = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    tasks, discovered = build_tasks(root, benchmark_names, largest_first=not args.small_first)
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
        "class_shuffle_method": "none" if args.no_class_shift else "shift",
        "softmax_temperature": args.softmax_temp,
        "average_logits": not args.no_average_logits,
        "use_amp": not args.no_amp,
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
    for task in tasks:
        task_queue.put(task)
    for _ in range(args.workers):
        task_queue.put(None)

    worker_csv_paths: List[Path] = []
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)
        process = mp.Process(
            target=worker_main,
            args=(worker_id, gpu_ids[worker_id], task_queue, str(worker_csv), dict(model_kwargs), args.verbose),
            daemon=False,
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    dfs: List[pd.DataFrame] = []
    for worker_csv in worker_csv_paths:
        if worker_csv.exists():
            try:
                dfs.append(pd.read_csv(worker_csv))
            except pd.errors.EmptyDataError:
                continue

    columns = list(ResultRow.__annotations__.keys())
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=columns)
    all_csv = out_dir / "all_classification_results.csv"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(out_dir / "summary.txt", all_df, sum(discovered.values()), wall_seconds)

    for benchmark in benchmark_names:
        benchmark_dir = out_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_df = all_df[all_df["benchmark"] == benchmark].copy() if len(all_df) else pd.DataFrame(columns=columns)
        benchmark_df.to_csv(benchmark_dir / "all_classification_results.csv", index=False)
        write_summary(benchmark_dir / "summary.txt", benchmark_df, discovered.get(benchmark, 0), wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {out_dir / 'summary.txt'}")
    for benchmark in benchmark_names:
        print(f"{benchmark}: {out_dir / benchmark / 'summary.txt'}")


if __name__ == "__main__":
    main()
