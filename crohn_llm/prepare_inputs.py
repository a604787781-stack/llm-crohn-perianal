# -*- coding: utf-8 -*-
"""Build model inputs from raw record sections: de-identification, deletion of CD and IBD-drug sentences,
and assembly of the six labelled sections.

Input: a JSON list of admissions with fields episode_id, patient_id, cd_label, hpi, pmh, discharge,
colonoscopy, mr, ct (the last three are lists of report texts).
Output: JSONL in the format expected by run_experiment.py (episode_id, patient_id, cd_label, prompt_text, meta).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from deidentify import mask_for_model, tidy_narrative


def join_reports(texts: list[str]) -> str:
    chunks = []
    for j, t in enumerate(texts, 1):
        chunks.append(f"--- 报告{j} ---\n{t.strip()}")
    return "\n\n".join(chunks)


def mask_and_join_reports(texts: list[str]) -> str:
    """逐份脱敏后再编号：整句删除不会吞掉报告分隔行，删空的报告直接略去，编号保持连续。"""
    return join_reports([m for m in (mask_for_model(t, report=True) for t in texts) if m.strip()])


def build_prompt_text(
    hpi: str, pmh: str, disc_cond: str, colo: str, mr: str, ct: str
) -> str:
    parts = [
        "【现病史】\n" + (hpi or "（缺）"),
        "【既往史】\n" + (pmh or "（缺）"),
        "【出院情况】\n" + (disc_cond or "（缺）"),
        "【肠镜报告】\n" + (colo or "（缺）"),
        "【MR报告】\n" + (mr or "（缺）"),
        "【CT/CTE报告】\n" + (ct or "（缺）"),
    ]
    return "\n\n".join(parts)


def build_episode(raw: dict) -> dict:
    hpi = tidy_narrative(mask_for_model(raw.get("hpi", "")))
    pmh = tidy_narrative(mask_for_model(raw.get("pmh", "")))
    disc = tidy_narrative(mask_for_model(raw.get("discharge", "")))
    colo = mask_and_join_reports(raw.get("colonoscopy", []))
    mr = mask_and_join_reports(raw.get("mr", []))
    ct = mask_and_join_reports(raw.get("ct", []))
    meta = {
        "n_colonoscopy": len(raw.get("colonoscopy", [])), "n_mr": len(raw.get("mr", [])), "n_ct": len(raw.get("ct", [])),
        "has_hpi": int(bool(hpi)), "has_pmh": int(bool(pmh)), "has_discharge_condition": int(bool(disc)),
        "has_colonoscopy": int(bool(colo)), "has_mr": int(bool(mr)), "has_ct": int(bool(ct)),
    }
    return {
        "episode_id": raw["episode_id"],
        "patient_id": raw["patient_id"],
        "cd_label": int(raw["cd_label"]),
        "prompt_text": build_prompt_text(hpi, pmh, disc, colo, mr, ct),
        "meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    raws = json.loads(args.raw.read_text(encoding="utf-8"))
    rows = [build_episode(r) for r in raws]
    by = defaultdict(list)
    for r in rows:
        by[r["patient_id"]].append(r)
    for items in by.values():
        for i, r in enumerate(items, 1):
            r["meta"].update(admission_index=i, n_admissions_patient=len(items), is_first_admission=int(i == 1))
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
