#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic multi-GPU benchmark runner for TabICLv2 (AMD/ROCm friendly).
Supports reading SINGLE CSV files and auto-splitting them 80/20 for train/test.

Key features:
- Dynamic scheduling (work stealing) across GPUs.
- Auto 8:2 train_test_split internally.
- Stratified split for Classification (with fallback to random split if classes are too small).
- Supports BOTH Classification and Regression tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score, mean_squared_error, mean_absolute_error
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# -----------------------------
# Helpers: dataset discovery
# -----------------------------

TARGET_CANDIDATES = [
    "target", "label", "class", "y",
    "TARGET", "Label", "Class", "Y",
]

def sanitize_dataset_id(path: Path) -> str:
    # 兼容原有的 OpenML-ID 格式，或者直接使用文件名（不含 .csv）
    m = re.search(r"(OpenML-ID-\d+)", str(path))
    return m.group(1) if m else path.stem

# -----------------------------
# Dataset skipping & Prioritization
# -----------------------------
SKIP_DATASETS = {
    "Smoking_and_Drinking_Dataset_with_body_signal",
    "Data_Science_for_Good_Kiva_Crowdfunding",
    "CDC_Diabetes_Health_Indicators",
    "walking-activity",
    "Rain_in_Australia",
    "accelerometer",
}

PRIORITY_DATASETS = [
    "customer_satisfaction_in_airline",
    "diabetes_130-us_hospitals",
    "dabetes_130-us_hospitals",
    "Credit_c",
    "SDSS17",
    "volkert",
]
_PRIORITY_INDEX = {name: i for i, name in enumerate(PRIORITY_DATASETS)}

def should_skip_dataset(path: Path) -> bool:
    ds_dir = path.parent.name
    ds_id = sanitize_dataset_id(path)
    return (ds_dir in SKIP_DATASETS) or (ds_id in SKIP_DATASETS)

def dataset_sort_key(path: Path):
    ds_dir = path.parent.name
    ds_id = sanitize_dataset_id(path)
    if ds_dir in _PRIORITY_INDEX:
        return (0, _PRIORITY_INDEX[ds_dir], ds_dir, str(path))
    if ds_id in _PRIORITY_INDEX:
        return (0, _PRIORITY_INDEX[ds_id], ds_id, str(path))
    return (1, 10**9, ds_id, str(path))

def find_datasets(root: Path) -> List[Path]:
    datasets: List[Path] = []
    for csv_path in root.rglob("*.csv"):
        # 过滤掉输出的 summary 或 worker 结果文件，避免误读
        if "tabicl_results" in csv_path.name or "worker_" in csv_path.name:
            continue
        # 兼容老逻辑：如果是 _test.csv 就跳过（假设老数据有 _train.csv 和 _test.csv）
        if csv_path.name.endswith("_test.csv"):
            continue
        if should_skip_dataset(csv_path):
            continue
        datasets.append(csv_path)
    return sorted(datasets, key=lambda x: dataset_sort_key(x))

def infer_target_column_single(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    # 如果没找到候选词，默认使用最后一列作为 Target
    return df.columns[-1]

def _normalize_local_ckpt_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    mp = Path(model_path).expanduser()
    try:
        mp = mp.resolve()
    except Exception:
        pass
    if not mp.exists():
        raise FileNotFoundError(f"Local checkpoint not found: {mp}")
    return str(mp)

def _default_all_out(out_dir: Path) -> Path:
    return out_dir / "tabicl_results.ALL.csv"

def _default_summary_txt(out_dir: Path) -> Path:
    return out_dir / "tabicl_results.summary.txt"

def _fmt_hms(seconds: float) -> str:
    if seconds is None:
        return ""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"

# -----------------------------
# Result schema
# -----------------------------

@dataclass
class ResultRow:
    dataset_id: str
    n_train: int
    n_test: int
    n_features: int
    n_classes: Optional[int]
    accuracy: Optional[float]
    f1_weighted: Optional[float]
    logloss: Optional[float]
    r2: Optional[float]           
    rmse: Optional[float]         
    mae: Optional[float]          
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]

# -----------------------------
# Core evaluation
# -----------------------------

def run_one_dataset_with_clf(
    clf,
    csv_path: Path,
    task: str,
) -> ResultRow:
    dataset_id = sanitize_dataset_id(csv_path)

    try:
        df = pd.read_csv(csv_path)
        target_col = infer_target_column_single(df)

        # 剔除 Target 列中的空值，防止切分或训练时报错
        df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 动态 8:2 切分
        if task == "classification":
            try:
                # 分类任务优先尝试分层抽样 (stratify)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                # 如果某个类样本太少导致分层失败，退回到随机抽样
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        else:
            # 回归任务直接随机抽样
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0

        t1 = time.time()
        if task == "classification":
            try:
                proba = clf.predict_proba(X_test)
                y_pred = clf.classes_[np.argmax(proba, axis=1)]
                ll = log_loss(y_test, proba, labels=clf.classes_)
            except Exception:
                y_pred = clf.predict(X_test)
                proba = None
                ll = None
        else:
            y_pred = clf.predict(X_test)
            proba = None
            ll = None
        pred_s = time.time() - t1

        acc = f1w = ll_val = r2 = rmse = mae = None
        
        if task == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1w = f1_score(y_test, y_pred, average="weighted")
            ll_val = float(ll) if ll is not None else None
            n_classes = getattr(clf, "n_classes_", int(y_train.nunique()))
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            n_classes = None

        return ResultRow(
            dataset_id=dataset_id,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            n_classes=n_classes,
            accuracy=float(acc) if acc is not None else None,
            f1_weighted=float(f1w) if f1w is not None else None,
            logloss=ll_val,
            r2=float(r2) if r2 is not None else None,
            rmse=float(rmse) if rmse is not None else None,
            mae=float(mae) if mae is not None else None,
            fit_seconds=float(fit_s),
            predict_seconds=float(pred_s),
            status="ok",
            error=None,
        )

    except Exception as e:
        return ResultRow(
            dataset_id=dataset_id,
            n_train=0,
            n_test=0,
            n_features=0,
            n_classes=None,
            accuracy=None,
            f1_weighted=None,
            logloss=None,
            r2=None,
            rmse=None,
            mae=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(e).__name__}: {e}",
        )

# -----------------------------
# Worker process
# -----------------------------

def worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue,
    out_csv: str,
    clf_kwargs: Dict,
    task: str,
    verbose: bool,
):
    try:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        if task == "classification":
            from tabicl import TabICLClassifier as TabICLModel
        else:
            from tabicl import TabICLRegressor as TabICLModel

        if not clf_kwargs.get("device"):
            clf_kwargs["device"] = "cuda:0"

        clf = TabICLModel(**clf_kwargs)

        rows: List[ResultRow] = []
        while True:
            item = task_queue.get()
            if item is None:
                break

            csv_path = item
            row = run_one_dataset_with_clf(clf, Path(csv_path), task)
            rows.append(row)

            if verbose:
                metric_str = f"acc={row.accuracy}" if task == "classification" else f"r2={row.r2}"
                print(f"[worker {worker_id} | gpu {gpu_id}] [{row.status}] {row.dataset_id} {metric_str}")

        df = pd.DataFrame([asdict(r) for r in rows])
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    except Exception:
        err = traceback.format_exc()
        crash_df = pd.DataFrame([{
            "dataset_id": f"__WORKER_CRASH__{worker_id}",
            "n_train": 0,
            "n_test": 0,
            "n_features": 0,
            "n_classes": None,
            "accuracy": None,
            "f1_weighted": None,
            "logloss": None,
            "r2": None,
            "rmse": None,
            "mae": None,
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "status": "fail",
            "error": err,
        }])
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        crash_df.to_csv(out_csv, index=False)
        if verbose:
            print(f"[worker {worker_id}] CRASHED:\n{err}")

# -----------------------------
# Summary writer
# -----------------------------

def write_summary_txt(
    out_txt: Path,
    root: Path,
    task: str,
    discovered_datasets: int,
    processed_datasets: int,
    failed_ids: List[str],
    avg_metric: Optional[float],
    topn_avgs: Dict[int, float],
    wall_seconds: Optional[float] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
):
    lines: List[str] = []
    lines.append(f"root: {root}")
    lines.append(f"task: {task}")
    lines.append(f"discovered_datasets: {discovered_datasets}")
    lines.append(f"processed_datasets: {processed_datasets}")

    if started_at is not None:
        lines.append(f"started_at: {started_at}")
    if finished_at is not None:
        lines.append(f"finished_at: {finished_at}")
    if wall_seconds is not None:
        lines.append(f"wall_seconds: {wall_seconds:.3f}")
        lines.append(f"wall_time_hms: {_fmt_hms(wall_seconds)}")

    lines.append(f"failed_count: {len(failed_ids)}")
    if failed_ids:
        lines.append("failed_datasets: " + ", ".join(failed_ids))
    else:
        lines.append("failed_datasets: (none)")

    main_metric = "accuracy" if task == "classification" else "r2"
    
    if avg_metric is None:
        lines.append(f"avg_{main_metric}_ok: (none)")
    else:
        lines.append(f"avg_{main_metric}_ok: {avg_metric:.6f}")

    for n in (27, 63, 154):
        if n in topn_avgs:
            lines.append(f"avg_{main_metric}_ok_top_{n}: {topn_avgs[n]:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing .csv files")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-worker CSVs and merged results")
    ap.add_argument("--task", type=str, default="classification", choices=["classification", "regression"], help="Task type to benchmark")
    ap.add_argument("--all-out", default=None, help="Path to merged ALL CSV")
    ap.add_argument("--summary-txt", default=None, help="Path to global summary txt")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker processes (usually #GPUs)")
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU ids to use (default: 0..workers-1)")
    
    ap.add_argument("--checkpoint-version", type=str, default=None, help="TabICL pretrained checkpoint")
    ap.add_argument("--model-path", type=str, default=None, help="Path to local TabICL checkpoint (.ckpt).")

    # TabICL common params
    ap.add_argument("--device", default="cuda:0", help='Device string in workers')
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-estimators", type=int, default=32)
    ap.add_argument("--norm-methods", default="none,power")
    ap.add_argument("--feat-shuffle", default="latin")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--kv-cache", action="store_true", help="Enable KV caching to speed up inference")
    ap.add_argument("--verbose", action="store_true")
    
    # Classification-only params
    ap.add_argument("--no-class-shift", action="store_true", help="(Classification only) Disable class shift")
    ap.add_argument("--softmax-temp", type=float, default=0.9, help="(Classification only) Softmax temperature")
    ap.add_argument("--no-average-logits", action="store_true", help="(Classification only) Average probabilities instead of logits")

    args = ap.parse_args()

    run_start_ts = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_start_ts))

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_out = Path(args.all_out) if args.all_out else _default_all_out(out_dir)
    summary_txt = Path(args.summary_txt) if args.summary_txt else _default_summary_txt(out_dir)

    root = Path(args.root)
    datasets = find_datasets(root)
    discovered_datasets = len(datasets)

    if discovered_datasets == 0:
        empty_df = pd.DataFrame(columns=[f.name for f in ResultRow.__dataclass_fields__.values()])
        all_out.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(all_out, index=False)

        run_end_ts = time.time()
        finished_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_end_ts))
        
        write_summary_txt(
            out_txt=summary_txt,
            root=root,
            task=args.task,
            discovered_datasets=0,
            processed_datasets=0,
            failed_ids=[],
            avg_metric=None,
            topn_avgs={},
            wall_seconds=(run_end_ts - run_start_ts),
            started_at=started_at,
            finished_at=finished_at,
        )
        print("No datasets found. Wrote empty outputs.")
        return

    workers = int(args.workers)
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(workers))

    norm_methods = [x.strip() for x in args.norm_methods.split(",") if x.strip()]
    model_path = _normalize_local_ckpt_path(args.model_path)

    model_kwargs: Dict = dict(
        n_estimators=args.n_estimators,
        norm_methods=norm_methods,
        feat_shuffle_method=args.feat_shuffle,
        use_amp=not args.no_amp,
        batch_size=args.batch_size,
        device=args.device,
        random_state=args.random_state,
        verbose=False,
    )
    
    if hasattr(args, 'kv_cache') and args.kv_cache:
        model_kwargs["kv_cache"] = True
    if args.checkpoint_version:
        model_kwargs["checkpoint_version"] = args.checkpoint_version
    if model_path is not None:
        model_kwargs["model_path"] = model_path
        model_kwargs["allow_auto_download"] = False

    if args.task == "classification":
        model_kwargs.update(dict(
            class_shift=not args.no_class_shift,
            softmax_temperature=args.softmax_temp,
            average_logits=not args.no_average_logits,
        ))

    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    task_queue: mp.Queue = mp.Queue()
    for csv_path in datasets:
        task_queue.put(str(csv_path))
    for _ in range(workers):
        task_queue.put(None)

    procs: List[mp.Process] = []
    worker_csv_paths: List[Path] = []
    for wid in range(workers):
        w_csv = out_dir / f"worker_{wid}.csv"
        worker_csv_paths.append(w_csv)
        p = mp.Process(
            target=worker_main,
            args=(wid, gpu_ids[wid], task_queue, str(w_csv), dict(model_kwargs), args.task, args.verbose),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    dfs: List[pd.DataFrame] = []
    for w_csv in worker_csv_paths:
        if w_csv.exists():
            try:
                dfs.append(pd.read_csv(w_csv))
            except Exception:
                continue

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=[f.name for f in ResultRow.__dataclass_fields__.values()])

    all_out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(all_out, index=False)

    processed_datasets = int(len(all_df))
    main_metric = "accuracy" if args.task == "classification" else "r2"

    if len(all_df):
        ok_df = all_df[(all_df["status"] == "ok")].copy()
    else:
        ok_df = pd.DataFrame()

    if len(ok_df) > 0 and main_metric in ok_df.columns:
        ok_df = ok_df[ok_df[main_metric].notna()]
        avg_metric = float(ok_df[main_metric].mean()) if len(ok_df) > 0 else None
    else:
        avg_metric = None

    cutoffs = (27, 63, 154)
    topn_avgs: Dict[int, float] = {}
    if len(ok_df) > 0:
        ok_sorted = ok_df.sort_values(main_metric, ascending=False, kind="mergesort")
        ok_count = len(ok_sorted)
        for n in cutoffs:
            if ok_count >= n:
                topn_avgs[n] = float(ok_sorted.head(n)[main_metric].mean())

    failed_ids: List[str] = []
    if len(all_df):
        failed_ids = (
            all_df.loc[all_df["status"] == "fail", "dataset_id"]
            .dropna()
            .astype(str)
            .tolist()
        )
        failed_ids = sorted(set(failed_ids))

    run_end_ts = time.time()
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_end_ts))
    wall_seconds = run_end_ts - run_start_ts

    write_summary_txt(
        out_txt=summary_txt,
        root=root,
        task=args.task,
        discovered_datasets=discovered_datasets,
        processed_datasets=processed_datasets,
        failed_ids=failed_ids,
        avg_metric=avg_metric,
        topn_avgs=topn_avgs,
        wall_seconds=wall_seconds,
        started_at=started_at,
        finished_at=finished_at,
    )

    print("\nSaved per-worker CSVs to:", str(out_dir))
    print("Saved merged ALL CSV to:", str(all_out))
    print("Saved summary TXT to:", str(summary_txt))
    print(f"\nTotal wall time: {wall_seconds:.3f}s ({_fmt_hms(wall_seconds)})")
    print(f"\nTabICL {args.task.capitalize()} kwargs:")
    print(json.dumps(model_kwargs, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()