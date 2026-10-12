# -*- coding: utf-8 -*-
"""训练阶段配置选择（SAP v1.1 第 5 节）。

- 主配置：训练评估集 AUROC 最高；与最高者差 < 0.005 时取单例平均总 token 更少者；仍并列取失败率更低者。
- 各模型最佳提示：同一规则在该模型 5 个提示中选择。
- 失败（无有效 probability 与 cd）按 SAP 4.2：AUROC 中 probability 记 50；二分类计为误判。
- probability 缺失但 cd 有效时，连续得分取 cd × 100（记录例数）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from paths import OUTPUTS
from prompt import PROMPT_VERSIONS

D = OUTPUTS / "formal" / "train"
OUT = D.parent / "train_selection.json"
MODELS = ["gpt55", "claudesonnet55", "dsv4pro", "dsflash", "qwen3_235b", "qwen3_14b", "qwen3_8b", "antangelmed"]
TIE = 0.005


def auroc(y: list[int], s: list[float]) -> float:
    pairs = sorted(zip(s, y))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    r1 = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]
    y, s, n_fail, n_cd_only, correct, tokens = [], [], 0, 0, 0, []
    tp = tn = fp = fn = 0
    for r in rows:
        g = int(r["gold"])
        p = r.get("probability")
        cd = r.get("pred")
        if p is None and cd not in (0, 1):
            n_fail += 1
            score, pred = 50.0, 1 - g
        elif p is None:
            n_cd_only += 1
            score, pred = 100.0 * cd, cd
        else:
            score, pred = float(p), int(float(p) >= 50)
        y.append(g)
        s.append(score)
        tp += g == 1 and pred == 1
        tn += g == 0 and pred == 0
        fp += g == 0 and pred == 1
        fn += g == 1 and pred == 0
        tokens.append(r["usage"].get("total_tokens") or 0)
    n = len(rows)
    return {
        "n": n,
        "auroc": round(auroc(y, s), 4),
        "sensitivity": round(tp / (tp + fn), 4),
        "specificity": round(tn / (tn + fp), 4),
        "accuracy": round((tp + tn) / n, 4),
        "n_failed": n_fail,
        "failure_rate": round(n_fail / n, 4),
        "n_cd_only": n_cd_only,
        "mean_total_tokens": round(sum(tokens) / n, 1),
        "file": path.name,
    }


def pick(cands: dict[str, dict]) -> str:
    best = max(v["auroc"] for v in cands.values())
    near = [k for k, v in cands.items() if best - v["auroc"] < TIE]
    return min(near, key=lambda k: (cands[k]["mean_total_tokens"], cands[k]["failure_rate"]))


def main():
    table: dict[str, dict] = {}
    missing = []
    for m in MODELS:
        for v in PROMPT_VERSIONS:
            f = D / f"{m}__{v}.json"
            if f.exists():
                table[f"{m}__{v}"] = evaluate(f)
            else:
                missing.append(f.name)
    if missing:
        raise SystemExit(f"尚有配置未完成：{missing}")
    main_cfg = pick(table)
    per_model = {m: pick({k: v for k, v in table.items() if k.startswith(m + "__")}) for m in MODELS}
    out = {
        "rule": "训练评估集 AUROC 最高；差 < 0.005 取平均总 token 更少者；再并列取失败率更低者",
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "main_config": main_cfg,
        "best_prompt_per_model": per_model,
        "table": table,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"主配置: {main_cfg}  {table[main_cfg]}")
    print("各模型最佳提示:", per_model)
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"{k:30s} AUROC={v['auroc']:.4f} sens={v['sensitivity']:.3f} spec={v['specificity']:.3f} "
              f"fail={v['n_failed']} tok={v['mean_total_tokens']:.0f}")
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
