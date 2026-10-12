# -*- coding: utf-8 -*-
"""测试集确证性与探索性分析（SAP v1.1 第 4、6–10 节）。

输入：outputs/formal/{train,test}/*.json、train_selection.json、冻结的关键词基线规则。
输出：outputs/formal/test_analysis.json 与 test_analysis.md。

- 失败（无有效 probability 与 cd）：AUROC 中 probability 记 50；二分类计为误判（SAP 4.2）。
- 95% CI：按患者整群 bootstrap 2000 次、百分位法、种子 2026；配对比较共用同一组重抽样（SAP 7）。
- 混合效应模型：本机无 R，按 SAP 6.3 预设的退化方案用患者整群 GEE（可交换相关、稳健标准误）。
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import statsmodels.api as sm
import statsmodels.formula.api as smf

from paths import DATA, OUTPUTS
from keyword_baseline import LEAK_GROUPS, MORPH_GROUPS, hits
from prompt import PROMPT_VERSIONS, fewshot_patient_ids

warnings.filterwarnings("ignore")

OUT = DATA
FORMAL = OUTPUTS / "formal"
B = 2000
SEED = 2026
MODELS = ["gpt55", "claudesonnet55", "dsv4pro", "dsflash", "qwen3_235b", "qwen3_14b", "qwen3_8b", "antangelmed"]
NAMES = {
    "gpt55": "GPT-5.5", "claudesonnet55": "Claude Sonnet 5.5", "dsv4pro": "DeepSeek-V4-Pro",
    "dsflash": "DeepSeek-Flash", "qwen3_235b": "Qwen3-235B", "qwen3_14b": "Qwen3-14B",
    "qwen3_8b": "Qwen3-8B", "antangelmed": "AntAngelMed", "kwB": "关键词基线 B",
}
CLOSED = ["gpt55", "claudesonnet55"]
OPEN = ["dsv4pro", "dsflash", "qwen3_235b", "qwen3_14b", "qwen3_8b"]
MEDICAL = ["antangelmed"]
REPEATS = {"gpt55__v0_zero": ["", "_rep2", "_rep3"], "antangelmed__v0_zero": ["", "_rep2", "_rep3"]}


# ---------- 数据 ----------

def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def load_episodes():
    fs = fewshot_patient_ids()
    tr = [e for e in load_jsonl(OUT / "train_episodes.jsonl") if e["patient_id"] not in fs]
    te = load_jsonl(OUT / "test_episodes.jsonl")
    pilot = {e["patient_id"] for e in json.loads((OUT / "pilot_episodes.json").read_text(encoding="utf-8"))}
    return tr, te, pilot


def score_rows(rows: list[dict], order: list[str]) -> dict:
    by = {r["episode_id"]: r for r in rows}
    s, pred, fail, cd_only, trunc, tok, sec, incons = [], [], [], [], [], [], [], []
    for eid in order:
        r = by[eid]
        g = int(r["gold"])
        p, cd = r.get("probability"), r.get("pred")
        if p is None and cd not in (0, 1):
            s.append(50.0); pred.append(1 - g); fail.append(1); cd_only.append(0)
        elif p is None:
            s.append(100.0 * cd); pred.append(cd); fail.append(0); cd_only.append(1)
        else:
            s.append(float(p)); pred.append(int(float(p) >= 50)); fail.append(0); cd_only.append(0)
            if cd in (0, 1) and cd != int(float(p) >= 50):
                incons.append(eid)
        trunc.append(int(bool(r.get("truncated"))))
        tok.append(r["usage"].get("total_tokens") or 0)
        sec.append(r.get("seconds") or 0)
    err = {}
    for eid in order:
        r = by[eid]
        if r.get("probability") is None and r.get("pred") not in (0, 1):
            e = r.get("api_error") or ""
            k = "内容审核拦截" if ("filtered" in e or "content" in e.lower()) else (e[:40] or f"解析失败:{r.get('parse_error','')[:30]}")
            err[k] = err.get(k, 0) + 1
    return dict(s=np.array(s), pred=np.array(pred), fail=np.array(fail), cd_only=int(sum(cd_only)),
                trunc=np.array(trunc), tok=np.array(tok), sec=np.array(sec), n_incons=len(incons), fail_reasons=err)


def load_config(split: str, name: str, order: list[str]) -> dict:
    d = json.loads((FORMAL / split / f"{name}.json").read_text(encoding="utf-8"))
    return score_rows(d["results"], order)


def keyword_scores(eps: list[dict], groups: dict) -> dict:
    s = np.array([len(hits(e["prompt_text"], groups)) for e in eps], dtype=float)
    n = len(eps)
    return dict(s=s, pred=(s > 0).astype(int), fail=np.zeros(n, int), cd_only=0, trunc=np.zeros(n, int),
                tok=np.zeros(n), sec=np.zeros(n), n_incons=0, fail_reasons={})


# ---------- 指标 ----------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def binary(y, pred) -> dict:
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    div = lambda a, b: a / b if b else np.nan
    sens, spec, ppv, npv = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp), div(tn, tn + fn)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                accuracy=(tp + tn) / len(y), balanced_accuracy=(sens + spec) / 2,
                f1=div(2 * ppv * sens, ppv + sens) if not np.isnan(ppv) else np.nan)


def logit_fit(y, x, offset_only=False):
    """返回校准截距（offset 模型）或 (截距, 斜率)；简单 IRLS。"""
    X = np.ones((len(y), 1)) if offset_only else np.column_stack([np.ones(len(y)), x])
    off = x if offset_only else np.zeros(len(y))
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        eta = X @ beta + off
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-10 * np.eye(X.shape[1]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def calibration(y, s) -> dict:
    p = np.clip(s, 1, 99) / 100
    lp = np.log(p / (1 - p))
    a = logit_fit(y, lp, offset_only=True)[0]
    _, slope = logit_fit(y, lp)
    return dict(brier=float(np.mean((s / 100 - y) ** 2)), cal_intercept=float(a), cal_slope=float(slope))


def full_metrics(y, c, with_cal=True) -> dict:
    m = {"auroc": auc(y, c["s"]), **binary(y, c["pred"])}
    if with_cal:
        m.update(calibration(y, c["s"]))
    return m


# ---------- bootstrap ----------

def boot_indices(pids: list[str], seed=SEED, b=B) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(pids))
    pos = {p: [] for p in uniq}
    for i, p in enumerate(pids):
        pos[p].append(i)
    groups = [np.array(pos[p]) for p in uniq]
    out = []
    for _ in range(b):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([groups[k] for k in pick]))
    return out


def ci(vals) -> list[float]:
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_metrics(y, c, idx, with_cal=False) -> dict:
    keys = ["auroc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_accuracy", "f1", "brier"]
    if with_cal:
        keys += ["cal_intercept", "cal_slope"]
    acc = {k: [] for k in keys}
    for ix in idx:
        yy, ss, pp = y[ix], c["s"][ix], c["pred"][ix]
        acc["auroc"].append(auc(yy, ss))
        bm = binary(yy, pp)
        for k in ["sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_accuracy", "f1"]:
            acc[k].append(bm[k])
        acc["brier"].append(np.mean((ss / 100 - yy) ** 2))
        if with_cal:
            cm = calibration(yy, ss)
            acc["cal_intercept"].append(cm["cal_intercept"]); acc["cal_slope"].append(cm["cal_slope"])
    return {k: ci(v) for k, v in acc.items()}


def boot_auc(y, s, idx) -> np.ndarray:
    return np.array([auc(y[ix], s[ix]) for ix in idx])


def paired_diff(y, sa, sb, idx, ba=None, bb=None) -> dict:
    ba = boot_auc(y, sa, idx) if ba is None else ba
    bb = boot_auc(y, sb, idx) if bb is None else bb
    d = ba - bb
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(auc_a=auc(y, sa), auc_b=auc(y, sb), diff=auc(y, sa) - auc(y, sb), ci=ci(d),
                p_boot=float(max(p, 1 / B)))


def delong(y, sa, sb) -> float:
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(s):
        x, z = s[pos], s[neg]
        v10 = np.array([((xi > z) + 0.5 * (xi == z)).mean() for xi in x])
        v01 = np.array([((x > zj) + 0.5 * (x == zj)).mean() for zj in z])
        return v10, v01

    a10, a01 = comps(sa); b10, b01 = comps(sb)
    s10 = np.cov(np.vstack([a10, b10])); s01 = np.cov(np.vstack([a01, b01]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (a10.mean() - b10.mean()) / np.sqrt(var) if var > 0 else 0.0
    return float(2 * norm.sf(abs(z)))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    k = len(ps)
    adj = np.empty(k)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (k - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ---------- 稳定性 ----------

def fleiss_kappa(mat: np.ndarray) -> float:
    n = mat.shape[1]
    counts = np.column_stack([(mat == 0).sum(1), (mat == 1).sum(1)])
    pi = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    pj = counts.sum(0) / counts.sum()
    pe = (pj ** 2).sum()
    return float((pi.mean() - pe) / (1 - pe)) if pe < 1 else 1.0


def icc21(mat: np.ndarray) -> float:
    n, k = mat.shape
    gm = mat.mean()
    msr = k * ((mat.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((mat.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((mat - mat.mean(1, keepdims=True) - mat.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


# ---------- GEE ----------

def gee_fit(df: pd.DataFrame, formula: str):
    mod = smf.gee(formula, groups="patient", data=df, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable())
    return mod.fit()


def contrast(res, weights: dict[str, float]) -> dict:
    names = list(res.params.index)
    L = np.zeros(len(names))
    for k, w in weights.items():
        L[names.index(k)] += w
    est = float(L @ res.params.values)
    se = float(np.sqrt(L @ res.cov_params().values @ L))
    z = est / se
    return dict(log_or=est, or_=float(np.exp(est)), ci=[float(np.exp(est - 1.96 * se)), float(np.exp(est + 1.96 * se))],
                p=float(2 * norm.sf(abs(z))))


def mterm(m):
    return f'C(model, Treatment("gpt55"))[T.{m}]'


def pterm(p):
    return f'C(prompt, Treatment("v0_zero"))[T.{p}]'


def model_weights(models: list[str], w: float) -> dict:
    return {mterm(m): w / len(models) for m in models if m != "gpt55"}


def add(d1: dict, d2: dict, sign=1.0) -> dict:
    out = dict(d1)
    for k, v in d2.items():
        out[k] = out.get(k, 0) + sign * v
    return out


# ---------- 主程序 ----------

def r4(x):
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def fingerprint(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: p.name):
        h.update(f.name.encode()); h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


def main():
    sel = json.loads((FORMAL / "train_selection.json").read_text(encoding="utf-8"))
    main_cfg = sel["main_config"]
    best = sel["best_prompt_per_model"]
    tr_eps, te_eps, pilot = load_episodes()
    te_order = [e["episode_id"] for e in te_eps]
    tr_order = [e["episode_id"] for e in tr_eps]
    y = np.array([int(e["cd_label"]) for e in te_eps])
    pids = [e["patient_id"] for e in te_eps]
    idx = boot_indices(pids)

    test_files = sorted(p for p in (FORMAL / "test").glob("*__*.json"))
    configs = {f"{m}__{v}": load_config("test", f"{m}__{v}", te_order) for m in MODELS for v in PROMPT_VERSIONS}
    kwB = keyword_scores(te_eps, {**LEAK_GROUPS, **MORPH_GROUPS})
    kwA = keyword_scores(te_eps, LEAK_GROUPS)

    # ---- 40 配置 + 基线：全部指标与 CI ----
    boot_cache = {}
    table = {}
    for name, c in {**configs, "kwB": kwB, "kwA": kwA}.items():
        is_kw = name.startswith("kw")
        m = full_metrics(y, c, with_cal=not is_kw)
        focus = name == main_cfg or name in best.values()
        bci = boot_metrics(y, c, idx, with_cal=focus and not is_kw)
        boot_cache[name] = boot_auc(y, c["s"], idx)
        n = len(y)
        table[name] = {
            **m, "ci": bci, "n": n, "n_failed": int(c["fail"].sum()), "failure_rate": c["fail"].mean(),
            "flag_fail_gt5pct": bool(c["fail"].mean() > 0.05), "fail_reasons": c["fail_reasons"],
            "n_cd_only": c["cd_only"], "n_cd_prob_inconsistent": c["n_incons"],
            "mean_total_tokens": float(c["tok"].mean()), "mean_seconds": float(c["sec"].mean()),
        }
        if is_kw:
            for k in ["brier"]:
                table[name]["ci"].pop(k, None)

    # ---- H1 ----
    h1 = paired_diff(y, configs[main_cfg]["s"], kwB["s"], idx, boot_cache[main_cfg], boot_cache["kwB"])
    h1["delong_p"] = delong(y, configs[main_cfg]["s"], kwB["s"])
    h1["main_auc_ci"] = ci(boot_cache[main_cfg])
    h1["kwB_auc_ci"] = ci(boot_cache["kwB"])
    h1["supported"] = bool(h1["ci"][0] > 0)

    # ---- F1：28 对 ----
    f1 = []
    for a, b in combinations(MODELS, 2):
        ca, cb = best[a], best[b]
        r = paired_diff(y, configs[ca]["s"], configs[cb]["s"], idx, boot_cache[ca], boot_cache[cb])
        f1.append({"a": ca, "b": cb, **r})
    for r, p in zip(f1, holm([r["p_boot"] for r in f1])):
        r["p_holm"] = p

    # ---- F5：各模型最佳 vs 基线 B ----
    f5 = []
    for m in MODELS:
        c = best[m]
        r = paired_diff(y, configs[c]["s"], kwB["s"], idx, boot_cache[c], boot_cache["kwB"])
        r["delong_p"] = delong(y, configs[c]["s"], kwB["s"])
        f5.append({"config": c, **r})
    for r, p in zip(f5, holm([r["p_boot"] for r in f5])):
        r["p_holm"] = p

    # ---- GEE：F2 / F3 / F4 ----
    long = []
    for m in MODELS:
        for v in PROMPT_VERSIONS:
            c = configs[f"{m}__{v}"]
            correct = (c["pred"] == y).astype(int)
            for i, eid in enumerate(te_order):
                long.append(dict(model=m, prompt=v, patient=pids[i], episode=eid, correct=int(correct[i])))
    df = pd.DataFrame(long)
    res = gee_fit(df, 'correct ~ C(model, Treatment("gpt55")) + C(prompt, Treatment("v0_zero"))')
    closed_w = model_weights(CLOSED, 1.0)
    open_w = model_weights(OPEN, 1.0)
    general_w = model_weights(CLOSED + OPEN, 1.0)
    f2 = {
        "closed_vs_open": contrast(res, add(closed_w, open_w, -1)),
        "medical_vs_general": contrast(res, add({mterm("antangelmed"): 1.0}, general_w, -1)),
    }
    for k, p in zip(f2, holm([v["p"] for v in f2.values()])):
        f2[k]["p_holm"] = p
    f3 = {
        "linear_trend_8b_14b_235b": contrast(res, {mterm("qwen3_8b"): -1.0, mterm("qwen3_235b"): 1.0}),
        "note": "有序得分 -1/0/1 的线性对比（14B 权重 0），OR 为 235B 相对 8B 的两端比较",
        "14b_vs_8b": contrast(res, {mterm("qwen3_14b"): 1.0, mterm("qwen3_8b"): -1.0}),
        "235b_vs_14b": contrast(res, {mterm("qwen3_235b"): 1.0, mterm("qwen3_14b"): -1.0}),
    }
    f4 = {v: contrast(res, {pterm(v): 1.0}) for v in PROMPT_VERSIONS[1:]}
    for k, p in zip(f4, holm([v["p"] for v in f4.values()])):
        f4[k]["p_holm"] = p
    model_or = {m: (contrast(res, {mterm(m): 1.0}) if m != "gpt55" else None) for m in MODELS}

    # ---- 稳定性 ----
    stability = {}
    for cfg, tags in REPEATS.items():
        runs = [load_config("test", cfg + t, te_order) for t in tags]
        P = np.column_stack([r["pred"] for r in runs])
        S = np.column_stack([r["s"] for r in runs])
        stability[cfg] = {
            "n_runs": len(runs),
            "auroc_each_run": [auc(y, r["s"]) for r in runs],
            "accuracy_each_run": [float((r["pred"] == y).mean()) for r in runs],
            "failed_each_run": [int(r["fail"].sum()) for r in runs],
            "binary_all_agree": float((P.min(1) == P.max(1)).mean()),
            "fleiss_kappa": fleiss_kappa(P),
            "prob_all_identical": float((S.min(1) == S.max(1)).mean()),
            "prob_icc21": icc21(S),
            "prob_max_range": float((S.max(1) - S.min(1)).max()),
        }

    # ---- 合并评估集（主配置与各模型最佳、基线 B）----
    all_eps = tr_eps + te_eps
    all_order = tr_order + te_order
    y_all = np.array([int(e["cd_label"]) for e in all_eps])
    pid_all = [e["patient_id"] for e in all_eps]
    idx_all = boot_indices(pid_all)

    def combined(cfg):
        a = load_config("train", cfg, tr_order); b = configs[cfg]
        return {k: (np.concatenate([a[k], b[k]]) if isinstance(a[k], np.ndarray) else a[k]) for k in a}

    comb = {c: combined(c) for c in sorted(set(best.values()) | {main_cfg})}
    comb["kwB"] = keyword_scores(all_eps, {**LEAK_GROUPS, **MORPH_GROUPS})

    # ---- 敏感性分析（主配置）----
    def sub_metrics(yv, c, mask, idx_src_pids):
        sub = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
        yy = yv[mask]
        ii = boot_indices([p for p, m in zip(idx_src_pids, mask) if m])
        m = full_metrics(yy, sub, with_cal=False)
        return {"n": int(mask.sum()), "n_pos": int(yy.sum()), "auroc": m["auroc"],
                "auroc_ci": ci(boot_auc(yy, sub["s"], ii)), "sensitivity": m["sensitivity"],
                "specificity": m["specificity"], "accuracy": m["accuracy"]}

    meta_te = [e["meta"] for e in te_eps]
    meta_all = [e["meta"] for e in all_eps]
    mc_te, mc_all = configs[main_cfg], comb[main_cfg]
    sens = {}
    for label, yv, c, meta, pp in [("test", y, mc_te, meta_te, pids), ("combined", y_all, mc_all, meta_all, pid_all)]:
        full = np.ones(len(yv), bool)
        s = {
            "S0_primary": sub_metrics(yv, c, full, pp),
            "S1_first_admission": sub_metrics(yv, c, np.array([m["is_first_admission"] == 1 for m in meta]), pp),
            "S2_complete_case": sub_metrics(yv, c, c["fail"] == 0, pp),
            "S3_drop_truncated": sub_metrics(yv, c, c["trunc"] == 0, pp),
        }
        if label == "combined":
            s["S4_untouched"] = sub_metrics(yv, c, np.array([p not in pilot for p in pp]), pp)
        sens[label] = s
    # S5：训练集 Youden 阈值
    tr_main = load_config("train", main_cfg, tr_order)
    y_tr = np.array([int(e["cd_label"]) for e in tr_eps])
    thr_best, j_best = 50.0, -1
    for t in np.unique(tr_main["s"]):
        pr = (tr_main["s"] >= t).astype(int)
        bm = binary(y_tr, pr)
        j = bm["sensitivity"] + bm["specificity"] - 1
        if j > j_best:
            thr_best, j_best = float(t), j
    sens["S5_youden"] = {
        "train_youden_threshold": thr_best, "train_youden_J": j_best,
        "test_threshold50": binary(y, mc_te["pred"]),
        "test_youden": binary(y, np.where(mc_te["fail"] == 1, 1 - y, (mc_te["s"] >= thr_best).astype(int))),
    }

    # ---- 亚组 ----
    def subgroup_block(yv, cdict, meta, pp, var, min_n=None):
        out = {}
        for level in (1, 0):
            mask = np.array([m[var] == level for m in meta])
            npos = int(yv[mask].sum()); nneg = int(mask.sum() - npos)
            row = {"n_pos": npos, "n_neg": nneg}
            if min_n and (npos < min_n or nneg < min_n):
                row["reported"] = False
            else:
                ii = boot_indices([p for p, m in zip(pp, mask) if m])
                for name, c in cdict.items():
                    ss = c["s"][mask]
                    row[name] = {"auroc": auc(yv[mask], ss), "ci": ci(boot_auc(yv[mask], ss, ii))}
                row["reported"] = True
            out["有" if level else "无"] = row
        return out

    te_sub_cfgs = {main_cfg: mc_te, "kwB": kwB}
    subgroups = {
        "test": {v: subgroup_block(y, te_sub_cfgs, meta_te, pids, v, min_n=10) for v in ["has_ct", "has_colonoscopy", "has_mr"]},
        "combined": {v: subgroup_block(y_all, comb, meta_all, pid_all, v) for v in ["has_ct", "has_colonoscopy", "has_mr"]},
    }
    # 交互（描述性）：合并集，8 个模型最佳配置
    inter = {}
    rows = []
    for m in MODELS:
        c = comb[best[m]]
        corr = (c["pred"] == y_all).astype(int)
        for i in range(len(y_all)):
            rows.append(dict(model=m, patient=pid_all[i], correct=int(corr[i]),
                             has_ct=meta_all[i]["has_ct"], has_colonoscopy=meta_all[i]["has_colonoscopy"],
                             has_mr=meta_all[i]["has_mr"]))
    dfi = pd.DataFrame(rows)
    for v in ["has_ct", "has_colonoscopy", "has_mr"]:
        r_full = gee_fit(dfi, f'correct ~ C(model, Treatment("gpt55")) * {v}')
        terms = [k for k in r_full.params.index if ":" in k]
        L = np.zeros((len(terms), len(r_full.params)))
        for i, t in enumerate(terms):
            L[i, list(r_full.params.index).index(t)] = 1
        w = r_full.wald_test(L, scalar=True)
        inter[v] = {"main_effect_or": float(np.exp(r_full.params[v])), "wald_chi2": float(w.statistic),
                    "df": len(terms), "p": float(w.pvalue)}

    # ---- 校准曲线（主配置，5 分位组）----
    s_main = mc_te["s"]
    q = pd.qcut(pd.Series(s_main).rank(method="first"), 5, labels=False)
    cal_curve = [{"bin": int(b), "n": int((q == b).sum()), "mean_pred": float(s_main[q == b].mean() / 100),
                  "obs_rate": float(y[q == b].mean())} for b in range(5)]

    # ---- 错误分析抽样 ----
    rng = np.random.default_rng(SEED)
    fp_ids = [te_order[i] for i in range(len(y)) if y[i] == 0 and mc_te["pred"][i] == 1]
    fn_ids = [te_order[i] for i in range(len(y)) if y[i] == 1 and mc_te["pred"][i] == 0]
    pick = lambda lst: sorted(rng.choice(lst, min(20, len(lst)), replace=False).tolist()) if lst else []
    err_sample = {"n_fp": len(fp_ids), "n_fn": len(fn_ids), "fp_sample": pick(fp_ids), "fn_sample": pick(fn_ids)}

    out = r4({
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "test_result_files": len(test_files), "test_result_fingerprint": fingerprint(test_files),
        "fingerprint_method": "按文件名排序，依次 sha256 更新（文件名 + 文件内容 sha256 摘要），取前 16 位",
        "n_test": len(y), "n_test_pos": int(y.sum()), "n_test_patients": len(set(pids)),
        "main_config": main_cfg, "best_prompt_per_model": best,
        "H1": h1, "F1_pairwise": f1, "F5_vs_baseline": f5,
        "gee": {"method": "GEE（Binomial，患者整群，可交换相关，稳健标准误）；参照 GPT-5.5 与 v0_zero",
                "n_obs": len(df), "model_or_vs_gpt55": model_or,
                "F2": f2, "F3": f3, "F4": f4},
        "stability": stability, "sensitivity": sens, "subgroups": subgroups, "subgroup_interaction": inter,
        "calibration_curve_main": cal_curve, "error_analysis_sample": err_sample, "table": table,
    })
    (FORMAL / "test_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(out)
    print("已写入 test_analysis.json / test_analysis.md")


def fmt_ci(v, c, nd=3):
    return f"{v:.{nd}f} ({c[0]:.{nd}f}–{c[1]:.{nd}f})" if v is not None and c else "—"


def write_md(o: dict):
    t = o["table"]
    L = [f"# 测试集分析结果（{o['computed_at']}）", "",
         f"测试集 {o['n_test']} 次住院 / {o['n_test_patients']} 名患者（阳性 {o['n_test_pos']}）；"
         f"结果文件 {o['test_result_files']} 个，合并指纹 {o['test_result_fingerprint']}。", ""]
    h = o["H1"]
    L += ["## H1 主要假设", "",
          f"- 主配置 {o['main_config']} AUROC = {fmt_ci(h['auc_a'], h['main_auc_ci'])}",
          f"- 关键词基线 B AUROC = {fmt_ci(h['auc_b'], h['kwB_auc_ci'])}",
          f"- 差值 = {fmt_ci(h['diff'], h['ci'])}；bootstrap p = {h['p_boot']:.4f}；DeLong p = {h['delong_p']:.2e}",
          f"- 结论：{'成立（CI 下限 > 0）' if h['supported'] else '不成立'}", ""]
    L += ["## 40 配置 + 基线", "", "| 配置 | AUROC (95% CI) | 灵敏度 | 特异度 | 准确率 | Brier | 失败 |", "|---|---|---|---|---|---|---|"]
    for k, v in sorted(t.items(), key=lambda kv: -kv[1]["auroc"]):
        brier = f"{v['brier']:.3f}" if v.get("brier") is not None else "—"
        flag = "（>5%）" if v["flag_fail_gt5pct"] else ""
        L.append(f"| {k} | {fmt_ci(v['auroc'], v['ci']['auroc'])} | {v['sensitivity']:.3f} | {v['specificity']:.3f} | "
                 f"{v['accuracy']:.3f} | {brier} | {v['n_failed']}{flag} |")
    L += ["", "## F5 各模型最佳 vs 基线 B（Holm）", "", "| 配置 | ΔAUROC (95% CI) | p_holm |", "|---|---|---|"]
    for r in o["F5_vs_baseline"]:
        L.append(f"| {r['config']} | {fmt_ci(r['diff'], r['ci'])} | {r['p_holm']:.4f} |")
    L += ["", "## F1 模型两两比较（Holm，仅列 p_holm < 0.05）", "", "| A | B | ΔAUROC (95% CI) | p_holm |", "|---|---|---|---|"]
    for r in sorted(o["F1_pairwise"], key=lambda r: r["p_holm"]):
        if r["p_holm"] < 0.05:
            L.append(f"| {r['a']} | {r['b']} | {fmt_ci(r['diff'], r['ci'])} | {r['p_holm']:.4f} |")
    g = o["gee"]
    L += ["", "## GEE（结局：判断正确）", ""]
    for fam in ["F2", "F4"]:
        for k, v in g[fam].items():
            L.append(f"- {fam} {k}: OR = {fmt_ci(v['or_'], v['ci'], 2)}，p = {v['p']:.4f}，p_holm = {v['p_holm']:.4f}")
    for k in ["linear_trend_8b_14b_235b", "14b_vs_8b", "235b_vs_14b"]:
        v = g["F3"][k]
        L.append(f"- F3 {k}: OR = {fmt_ci(v['or_'], v['ci'], 2)}，p = {v['p']:.4f}")
    L += ["", "## 稳定性（3 次运行）", ""]
    for k, v in o["stability"].items():
        L.append(f"- {k}: 二分类三次一致 {v['binary_all_agree']:.3f}，Fleiss κ = {v['fleiss_kappa']:.3f}，"
                 f"probability 完全相同 {v['prob_all_identical']:.3f}，ICC(2,1) = {v['prob_icc21']:.3f}，"
                 f"各次 AUROC {v['auroc_each_run']}")
    L += ["", "## 敏感性分析（主配置）", ""]
    for split in ["test", "combined"]:
        for k, v in o["sensitivity"][split].items():
            L.append(f"- {split} {k}: n={v['n']}（阳性 {v['n_pos']}），AUROC = {fmt_ci(v['auroc'], v['auroc_ci'])}")
    s5 = o["sensitivity"]["S5_youden"]
    L.append(f"- S5 训练集 Youden 阈值 = {s5['train_youden_threshold']}；测试集 阈值50 灵敏度/特异度 "
             f"{s5['test_threshold50']['sensitivity']:.3f}/{s5['test_threshold50']['specificity']:.3f}，Youden 阈值 "
             f"{s5['test_youden']['sensitivity']:.3f}/{s5['test_youden']['specificity']:.3f}")
    L += ["", "## 亚组（AUROC）", ""]
    for split in ["test", "combined"]:
        for var, blk in o["subgroups"][split].items():
            for lvl, row in blk.items():
                if not row.get("reported"):
                    L.append(f"- {split} {var}={lvl}：阳性 {row['n_pos']} / 阴性 {row['n_neg']}，样本不足不报告")
                    continue
                items = "；".join(f"{k} {fmt_ci(v['auroc'], v['ci'])}" for k, v in row.items() if isinstance(v, dict))
                L.append(f"- {split} {var}={lvl}（{row['n_pos']}/{row['n_neg']}）：{items}")
    for v, r in o["subgroup_interaction"].items():
        L.append(f"- 交互（合并集，8 模型最佳）{v}: χ² = {r['wald_chi2']:.2f}，df = {r['df']}，p = {r['p']:.4f}")
    e = o["error_analysis_sample"]
    L += ["", "## 错误分析抽样", "", f"- 假阳性 {e['n_fp']} 例，抽取 {len(e['fp_sample'])}；假阴性 {e['n_fn']} 例，抽取 {len(e['fn_sample'])}"]
    (FORMAL / "test_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
