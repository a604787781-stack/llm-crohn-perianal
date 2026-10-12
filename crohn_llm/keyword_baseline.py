# -*- coding: utf-8 -*-
"""关键词规则基线：量化病名语句删除后残留的诊断线索。

规则 A（泄露探针）：IBD 诊断名 / IBD 专用药物，命中任一即判 1。
规则 B（A + 形态学）：再加入克罗恩病典型内镜/影像描述。
连续得分 = 命中的不同关键词组数，用于计算 AUROC。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from paths import DATA, OUTPUTS

OUT_DIR = OUTPUTS
TRAIN_JSONL = DATA / "train_episodes.jsonl"

LEAK_GROUPS = {
    "IBD诊断名": r"炎症性肠病|炎性肠病|IBD",
    "氨基水杨酸": r"美沙拉[秦嗪]|柳氮磺吡啶|巴柳氮|奥沙拉秦|艾迪莎|颇得斯安|莎尔福|5-?ASA",
    "生物制剂": r"英夫利[西昔]|类克|阿达木|修美乐|乌司奴|喜达诺|维得利珠|安吉优|生物制剂|IFX|UST|ADA(?![A-Za-z])",
    "免疫抑制剂": r"硫唑嘌呤|依木兰|巯嘌呤|6-?MP|甲氨蝶呤|沙利度胺|他克莫司|环孢素",
    "肠内营养治疗": r"全肠内营养|EEN|肠内营养治疗",
}
MORPH_GROUPS = {
    "纵行溃疡": r"纵行溃疡|纵形溃疡",
    "鹅卵石/铺路石": r"鹅卵石|铺路石|卵石样",
    "节段/跳跃性": r"节段性|跳跃性|跳跃式|非连续性",
    "裂隙/阿弗他溃疡": r"裂隙样?溃疡|阿弗他|口疮样溃疡",
    "梳状征/靶征": r"梳[状样]征|靶征|分层强化",
    "肉芽肿": r"肉芽肿",
}


def hits(text: str, groups: dict[str, str]) -> list[str]:
    return [g for g, pat in groups.items() if re.search(pat, text)]


def metrics(pairs):
    tp = sum(g == 1 and p == 1 for g, p in pairs)
    tn = sum(g == 0 and p == 0 for g, p in pairs)
    fp = sum(g == 0 and p == 1 for g, p in pairs)
    fn = sum(g == 1 and p == 0 for g, p in pairs)
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    ppv = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else 0
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round((tp + tn) / n, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "f1": round(f1, 4),
        "balanced_acc": round((sens + spec) / 2, 4),
    }


def auroc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def evaluate(rows, groups):
    ep_labels, ep_scores, ep_pairs = [], [], []
    by = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        g = int(r["cd_label"])
        score = len(hits(r["prompt_text"], groups))
        pred = int(score > 0)
        ep_labels.append(g)
        ep_scores.append(score)
        ep_pairs.append((g, pred))
        b = by[r["patient_id"]]
        b[0] |= g
        b[1] |= pred
        b[2] = max(b[2], score)
    return {
        "episode": {**metrics(ep_pairs), "auroc": auroc(ep_labels, ep_scores)},
        "patient": {
            **metrics([(v[0], v[1]) for v in by.values()]),
            "auroc": auroc([v[0] for v in by.values()], [v[2] for v in by.values()]),
        },
    }


def main():
    rows = [json.loads(l) for l in TRAIN_JSONL.open(encoding="utf-8")]
    n_pos = sum(int(r["cd_label"]) == 1 for r in rows)
    n_neg = len(rows) - n_pos

    per_group = []
    for g, pat in {**LEAK_GROUPS, **MORPH_GROUPS}.items():
        p = sum(1 for r in rows if int(r["cd_label"]) == 1 and re.search(pat, r["prompt_text"]))
        q = sum(1 for r in rows if int(r["cd_label"]) == 0 and re.search(pat, r["prompt_text"]))
        per_group.append({
            "group": g,
            "type": "leak" if g in LEAK_GROUPS else "morph",
            "pos_hits": p,
            "pos_rate": round(p / n_pos, 4),
            "neg_hits": q,
            "neg_rate": round(q / n_neg, 4),
            "ppv": round(p / (p + q), 4) if p + q else None,
        })

    report = {
        "n_episodes": len(rows),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "rule_A_leak": evaluate(rows, LEAK_GROUPS),
        "rule_B_leak_plus_morph": evaluate(rows, {**LEAK_GROUPS, **MORPH_GROUPS}),
        "per_group": per_group,
    }
    (OUT_DIR / "keyword_baseline_train.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
