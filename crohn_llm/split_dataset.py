# -*- coding: utf-8 -*-
"""按患者划分 train/test（分层抽样），禁止同一患者跨集。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from paths import DATA

IN_JSONL = DATA / "episode_inputs.jsonl"
OUT_DIR = DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    rows = [json.loads(l) for l in IN_JSONL.open(encoding="utf-8")]
    # 患者级标签：任一次住院阳性 → 患者阳性（与金标准一致的实用规则）
    by_pid: dict[str, list] = defaultdict(list)
    for r in rows:
        by_pid[r["patient_id"]].append(r)

    patient_label = {}
    for pid, items in by_pid.items():
        patient_label[pid] = 1 if any(int(x["cd_label"]) == 1 for x in items) else 0

    pos = [p for p, y in patient_label.items() if y == 1]
    neg = [p for p, y in patient_label.items() if y == 0]
    rng = random.Random(args.seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split(ids: list[str]):
        n_test = max(1, int(round(len(ids) * args.test_ratio)))
        return ids[n_test:], ids[:n_test]

    train_pos, test_pos = split(pos)
    train_neg, test_neg = split(neg)
    train_pids = set(train_pos + train_neg)
    test_pids = set(test_pos + test_neg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "n_patients": len(patient_label),
        "n_train_patients": len(train_pids),
        "n_test_patients": len(test_pids),
        "train_pos": len(train_pos),
        "train_neg": len(train_neg),
        "test_pos": len(test_pos),
        "test_neg": len(test_neg),
        "train_patient_ids": sorted(train_pids),
        "test_patient_ids": sorted(test_pids),
    }
    (OUT_DIR / "split.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def write_subset(name: str, pids: set[str]):
        path = OUT_DIR / f"{name}_episodes.jsonl"
        n = 0
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                if r["patient_id"] in pids:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
        return n

    n_tr = write_subset("train", train_pids)
    n_te = write_subset("test", test_pids)

    # 患者级索引表
    with (OUT_DIR / "split_patients.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "patient_label", "split", "n_episodes"])
        w.writeheader()
        for pid in sorted(patient_label):
            w.writerow(
                {
                    "patient_id": pid,
                    "patient_label": patient_label[pid],
                    "split": "test" if pid in test_pids else "train",
                    "n_episodes": len(by_pid[pid]),
                }
            )

    print(
        f"患者 train={len(train_pids)} (pos {len(train_pos)}/neg {len(train_neg)}) "
        f"test={len(test_pids)} (pos {len(test_pos)}/neg {len(test_neg)})"
    )
    print(f"住院样本 train={n_tr} test={n_te}")
    print(f"写出: {OUT_DIR / 'split.json'}")


if __name__ == "__main__":
    main()
