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
from sklearn.model_selection import train_test_split


DEFAULT_BENCHMARKS = [
    "openml_cc18_csv=../limix/openml_cc18_csv",
    "tabarena_cls=dataset/tabarena/cls",
    "tabzilla_csv=../limix/tabzilla_csv",
    "talent_csv=../limix/talent_csv",
]
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


def sanitize_dataset_id(path: Path) -> str:
    match = re.search(r"(OpenML-ID-\d+)", str(path))
    if match:
        return match.group(1)
    stem = path.stem
    if stem.endswith("_train"):
        return stem[:-6]
    if stem.endswith("_test"):
        return stem[:-5]
    return stem


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in train_df.columns:
            return col
    extra = [col for col in train_df.columns if col not in test_df.columns]
    if len(extra) == 1:
        return extra[0]
    return train_df.columns[-1]


def infer_target_column_single(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[-1]


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def parse_benchmark_specs(root: Path, specs: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            name, rel_path = spec.split("=", 1)
            benchmark_name = name.strip()
            benchmark_path = Path(rel_path.strip())
        else:
            benchmark_path = root / spec.strip()
            benchmark_name = benchmark_path.name

        if not benchmark_path.is_absolute():
            benchmark_path = (root / benchmark_path).resolve()
        else:
            benchmark_path = benchmark_path.resolve()

        parsed.append((benchmark_name, benchmark_path))
    return parsed


def discover_benchmark_tasks(benchmark_dir: Path, benchmark_name: str, largest_first: bool) -> List[Tuple[str, str, str, str, int]]:
    paired_tasks: List[Tuple[str, str, str, str, int]] = []
    paired_train_paths: set[Path] = set()
    paired_test_paths: set[Path] = set()

    for train_csv in benchmark_dir.rglob("*_train.csv"):
        test_csv = train_csv.with_name(train_csv.name.replace("_train.csv", "_test.csv"))
        if not test_csv.exists():
            continue
        paired_train_paths.add(train_csv.resolve())
        paired_test_paths.add(test_csv.resolve())
        size_bytes = file_size(train_csv) + file_size(test_csv)
        paired_tasks.append((benchmark_name, "paired", str(train_csv), str(test_csv), size_bytes))

    single_tasks: List[Tuple[str, str, str, str, int]] = []
    for csv_path in benchmark_dir.rglob("*.csv"):
        resolved = csv_path.resolve()
        if resolved in paired_train_paths or resolved in paired_test_paths:
            continue
        if csv_path.name.endswith("_train.csv") or csv_path.name.endswith("_test.csv"):
            continue
        size_bytes = file_size(csv_path)
        single_tasks.append((benchmark_name, "split", str(csv_path), "", size_bytes))

    tasks = paired_tasks + single_tasks
    if largest_first:
        tasks.sort(key=lambda x: (-x[4], x[0], x[2]))
    else:
        tasks.sort(key=lambda x: x[2])
    return tasks


def build_tasks(root: Path, benchmark_specs: List[str], largest_first: bool) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int]]:
    per_benchmark: Dict[str, List[Tuple[str, str, str, str, int]]] = {}
    discovered: Dict[str, int] = {}

    for benchmark_name, benchmark_dir in parse_benchmark_specs(root, benchmark_specs):
        tasks = discover_benchmark_tasks(benchmark_dir, benchmark_name, largest_first) if benchmark_dir.exists() else []
        per_benchmark[benchmark_name] = tasks
        discovered[benchmark_name] = len(tasks)

    merged: List[Tuple[str, str, str, str]] = []
    benchmark_names = [name for name, _ in parse_benchmark_specs(root, benchmark_specs)]
    while True:
        batch: List[Tuple[str, str, str, str, int]] = []
        for benchmark_name in benchmark_names:
            if per_benchmark[benchmark_name]:
                batch.append(per_benchmark[benchmark_name].pop(0))
        if not batch:
            break
        if largest_first:
            batch.sort(key=lambda x: (-x[4], x[0], x[2]))
        for benchmark_name, split_kind, train_path, test_path, _ in batch:
            merged.append((benchmark_name, split_kind, train_path, test_path))
    return merged, discovered


def split_single_dataset(df: pd.DataFrame, random_state: int, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target_col = infer_target_column_single(df)
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if len(X) < 2:
        raise ValueError("Not enough valid rows after dropping missing target.")

    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_train_test(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Optional[pd.Series]]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target_col = infer_target_column(train_df, test_df)

    train_df = train_df.dropna(subset=[target_col])
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    if target_col in test_df.columns:
        test_df = test_df.dropna(subset=[target_col])
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
    else:
        X_test = test_df
        y_test = None
    return X_train, X_test, y_train, y_test


def evaluate_one_dataset(
    clf,
    benchmark: str,
    split_kind: str,
    train_path: Path,
    test_path: Optional[Path],
    test_size: float,
    random_state: int,
) -> ResultRow:
    dataset_id = sanitize_dataset_id(train_path)
    try:
        if split_kind == "paired":
            X_train, X_test, y_train, y_test = load_train_test(train_path, test_path if test_path is not None else train_path)
        else:
            df = pd.read_csv(train_path)
            X_train, X_test, y_train, y_test = split_single_dataset(df, random_state=random_state, test_size=test_size)

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        try:
            proba = clf.predict_proba(X_test)
            y_pred = clf.classes_[np.argmax(proba, axis=1)]
            ll = log_loss(y_test, proba, labels=clf.classes_)
        except Exception:
            y_pred = clf.predict(X_test)
            ll = None
        predict_seconds = time.time() - t1

        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        return ResultRow(
            benchmark=benchmark,
            dataset_id=dataset_id,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            n_classes=int(y_train.nunique()),
            accuracy=float(accuracy),
            f1_weighted=float(f1_weighted),
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
    task_items: List[Tuple[str, str, str, str]],
    worker_out_csv: str,
    model_kwargs: Dict,
    test_size: float,
    random_state: int,
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
        for item in task_items:
            benchmark, split_kind, train_path, test_path = item
            row = evaluate_one_dataset(
                clf,
                benchmark=benchmark,
                split_kind=split_kind,
                train_path=Path(train_path),
                test_path=Path(test_path) if test_path else None,
                test_size=test_size,
                random_state=random_state,
            )
            rows.append(row)

            if verbose:
                if row.status == "ok":
                    print(f"[worker {worker_id} | gpu {gpu_id}] [ok] {benchmark}/{row.dataset_id} acc={row.accuracy:.6f}")
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


def write_summary(summary_path: Path, result_df: pd.DataFrame, discovered_datasets: int, wall_seconds: float) -> None:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {discovered_datasets}",
        f"processed_datasets: {len(result_df)}",
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
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--out-dir", default="result/TabICLv2_official_classification")
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
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-class-shift", action="store_true")
    parser.add_argument("--no-average-logits", action="store_true")
    parser.add_argument("--small-first", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    try:
        root = root.resolve()
    except Exception:
        pass
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

    benchmark_specs = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    benchmark_names = [name for name, _ in parse_benchmark_specs(root, benchmark_specs)]
    tasks, discovered = build_tasks(root, benchmark_specs, largest_first=not args.small_first)
    if not tasks:
        raise FileNotFoundError("No classification datasets found in the configured benchmark directories.")

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
    per_worker_tasks: List[List[Tuple[str, str, str, str]]] = [[] for _ in range(args.workers)]
    for idx, task in enumerate(tasks):
        per_worker_tasks[idx % args.workers].append(task)

    worker_csv_paths: List[Path] = []
    processes: List[mp.Process] = []
    for worker_id in range(args.workers):
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        worker_csv_paths.append(worker_csv)
        process = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_ids[worker_id],
                per_worker_tasks[worker_id],
                str(worker_csv),
                dict(model_kwargs),
                args.test_size,
                args.random_state,
                args.verbose,
            ),
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
