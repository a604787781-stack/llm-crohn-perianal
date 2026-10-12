# -*- coding: utf-8 -*-
"""正式实验总控：8 模型 × 5 prompt，按 SAP v1.0 运行。

- 复用已冻结的 llm_client.run_one / 各后端，不改动其调用与解析逻辑。
- 每个模型一条通道，通道内 5 个 prompt 顺序运行；同平台通道共享「停止」信号。
- 余额不足 / 鉴权失败 / 连续失败时立即停止该平台，写入 status.json，等待充值后重跑本脚本即可断点续跑。
- 失败处理（SAP 3、4.2）：瞬时错误由 run_one 内部重试；整轮结束后对无有效输出的住院最多再补跑 2 轮。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import OUTPUTS  # noqa: E402
from prompt import PROMPT_VERSIONS, fewshot_patient_ids  # noqa: E402
from llm_client import (  # noqa: E402
    DeepSeekBackend,
    OpenAICompatBackend,
    _load_pool,
    load_dotenv,
    run_one,
    summarize,
)

OUT_ROOT = OUTPUTS / "formal"
MAX_CHARS = 12000
TEMPERATURE = 0.0
MAX_ATTEMPTS = 3
CONSECUTIVE_FAIL_HALT = 15

MODELS = {
    "gpt55": dict(platform="relay", model="gpt-5.5-2026-04-24", reasoning_effort="none", max_tokens=4096, workers=3),
    "claudesonnet55": dict(platform="relay", model="claude-sonnet-5-5", max_tokens=4096, workers=8),
    "dsv4pro": dict(platform="deepseek", model="deepseek-v4-pro", workers=4),
    "dsflash": dict(platform="deepseek", model="deepseek-flash", workers=4),
    "qwen3_235b": dict(platform="dashscope", model="qwen3-235b-a22b", max_tokens=4096, workers=3),
    "qwen3_14b": dict(platform="dashscope", model="qwen3-14b", max_tokens=4096, workers=3),
    "qwen3_8b": dict(platform="dashscope", model="qwen3-8b", max_tokens=4096, workers=3),
    "antangelmed": dict(platform="antling", model="AntAngelMed", max_tokens=16384, workers=5),
}

PLATFORM_NAMES = {"relay": "中转站（GPT/Claude）", "deepseek": "DeepSeek 官方", "dashscope": "千问原厂（阿里云百炼）", "antling": "蚂蚁百灵"}

HALT_PATTERN = re.compile(
    r"\b(?:401|402|403)\b|insufficient|balance|quota|余额|欠费|额度|arrearage|overdue|payment|billing|"
    r"invalid.?api.?key|unauthori[sz]ed|authentication|permission.?denied",
    re.I,
)


# 内容审核拦截是逐例问题，不计入连续失败
CONTENT_FILTER = re.compile(r"content management|content_filter|content policy|data_inspection|inappropriate", re.I)


def make_backend(cfg: dict):
    p = cfg["platform"]
    if p == "deepseek":
        return DeepSeekBackend(os.environ["DEEPSEEK_API_KEY"])
    if p == "relay":
        return OpenAICompatBackend(
            os.environ["RELAY_API_KEY"], os.environ.get("RELAY_BASE_URL", "https://quanzil.com/v1"),
            cfg.get("reasoning_effort", ""), max_tokens=cfg["max_tokens"],
        )
    if p == "dashscope":
        return OpenAICompatBackend(
            os.environ["DASHSCOPE_API_KEY"],
            os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "", max_tokens=cfg["max_tokens"],
        )
    if p == "antling":
        return OpenAICompatBackend(
            os.environ["ANTANGELMED_API_KEY"], "https://api.ant-ling.com/v1", "", max_tokens=cfg["max_tokens"]
        )
    raise ValueError(p)


def is_valid(row: dict) -> bool:
    return row.get("pred") in (0, 1) or row.get("probability") is not None


class Status:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"started": datetime.now().isoformat(timespec="seconds"), "platforms": {}, "configs": {}}

    def update(self, section: str, key: str, **kw):
        with self.lock:
            self.data[section].setdefault(key, {}).update(kw)
            self.data["updated"] = datetime.now().isoformat(timespec="seconds")
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ckpt(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                latest[r["episode_id"]] = r
    return latest


def run_config(key, cfg, version, episodes, out_dir, halt: threading.Event, status: Status, repeat_tag: str):
    name = f"{key}__{version}{repeat_tag}"
    ckpt = out_dir / f"{name}.ckpt.jsonl"
    final = out_dir / f"{name}.json"
    backend = make_backend(cfg)
    lock = threading.Lock()
    consecutive = [0]

    def needs(ep, latest):
        r = latest.get(ep["episode_id"])
        return r is None or (not is_valid(r) and r.get("attempt", 1) < MAX_ATTEMPTS)

    def work(ep, attempt):
        if halt.is_set():
            return
        row = run_one(backend, cfg["model"], ep, MAX_CHARS, False, version, TEMPERATURE)
        row["attempt"] = attempt
        row["model_key"] = key
        row["called_at"] = datetime.now().isoformat(timespec="seconds")
        err = row["api_error"]
        with lock:
            if err and HALT_PATTERN.search(err):
                if not halt.is_set():
                    halt.set()
                    status.update("platforms", cfg["platform"], state="halted", reason="余额不足或鉴权失败",
                                  error=err[:400], config=name,
                                  at=datetime.now().isoformat(timespec="seconds"))
                return
            if not (err and CONTENT_FILTER.search(err)):
                consecutive[0] = consecutive[0] + 1 if err else 0
            with ckpt.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if consecutive[0] >= CONSECUTIVE_FAIL_HALT and not halt.is_set():
                halt.set()
                status.update("platforms", cfg["platform"], state="halted", reason=f"连续 {CONSECUTIVE_FAIL_HALT} 次调用失败",
                              error=err[:400], config=name, at=datetime.now().isoformat(timespec="seconds"))

    t0 = time.time()
    for _ in range(MAX_ATTEMPTS):
        latest = load_ckpt(ckpt)
        todo = [ep for ep in episodes if needs(ep, latest)]
        if not todo or halt.is_set():
            break
        status.update("configs", name, state="running", done=len(episodes) - len(todo), n=len(episodes))
        with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
            list(ex.map(lambda ep: work(ep, latest.get(ep["episode_id"], {}).get("attempt", 0) + 1), todo))

    latest = load_ckpt(ckpt)
    done = sum(1 for ep in episodes if ep["episode_id"] in latest)
    if halt.is_set() and any(needs(ep, latest) for ep in episodes):
        status.update("configs", name, state="paused", done=done, n=len(episodes))
        return False

    results = [latest[ep["episode_id"]] for ep in episodes]
    n_fail = sum(not is_valid(r) for r in results)
    meta = {
        "split": out_dir.name,
        "prompt_version": version,
        "temperature": TEMPERATURE,
        "platform": cfg["platform"],
        "reasoning_effort": cfg.get("reasoning_effort") or None,
        "max_tokens": cfg.get("max_tokens"),
        "dropped_params": sorted(getattr(backend, "drop", [])),
        "n_failed_final": n_fail,
        "n_api_error_final": sum(bool(r["api_error"]) for r in results),
        "n_parse_error_final": sum((not r["api_error"]) and not is_valid(r) for r in results),
        "n_truncated_input": sum(bool(r["truncated"]) for r in results),
        "n_truncated_output": sum(r["usage"].get("finish_reason") == "length" for r in results),
        "total_reasoning_tokens": sum(r["usage"].get("reasoning_tokens") or 0 for r in results),
        "excluded_fewshot_patients": sorted(fewshot_patient_ids()),
        "repeat_tag": repeat_tag or None,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds_this_session": round(time.time() - t0, 1),
    }
    summary = summarize(results, cfg["model"], False, meta)
    final.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    status.update("configs", name, state="done", done=len(episodes), n=len(episodes), failed=n_fail,
                  accuracy=summary["accuracy"])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--prompts", nargs="*", default=PROMPT_VERSIONS)
    ap.add_argument("--repeat-tag", default="", help="测试集重复运行标记，如 _rep2")
    ap.add_argument("--smoke", type=int, default=0, help="只跑前 N 例，输出到 smoke_<split>，不计入正式结果")
    ap.add_argument("--status-name", default="status", help="并行多个进程时各用一个状态文件")
    args = ap.parse_args()

    load_dotenv()
    excluded = fewshot_patient_ids()
    episodes = [e for e in _load_pool(args.split) if e["patient_id"] not in excluded]
    out_dir = OUT_ROOT / args.split
    if args.smoke:
        episodes = episodes[: args.smoke]
        out_dir = OUT_ROOT / f"smoke_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / f"{args.status_name}{args.repeat_tag}.json")
    print(f"split={args.split} n={len(episodes)} pos={sum(int(e['cd_label']) for e in episodes)} "
          f"models={args.models} prompts={args.prompts}", flush=True)

    halts = {p: threading.Event() for p in PLATFORM_NAMES}
    for p in {MODELS[k]["platform"] for k in args.models}:
        status.update("platforms", p, state="running", name=PLATFORM_NAMES[p])

    def lane(key):
        cfg = MODELS[key]
        for v in args.prompts:
            if halts[cfg["platform"]].is_set():
                return
            try:
                ok = run_config(key, cfg, v, episodes, out_dir, halts[cfg["platform"]], status, args.repeat_tag)
            except Exception as e:
                halts[cfg["platform"]].set()
                status.update("platforms", cfg["platform"], state="halted", reason="程序异常",
                              error=f"{type(e).__name__}: {e}"[:400], config=f"{key}__{v}",
                              at=datetime.now().isoformat(timespec="seconds"))
                raise
            print(f"[{datetime.now():%H:%M:%S}] {key}__{v}{args.repeat_tag} {'完成' if ok else '暂停'}", flush=True)

    threads = [threading.Thread(target=lane, args=(k,), daemon=True) for k in args.models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for p in {MODELS[k]["platform"] for k in args.models}:
        if not halts[p].is_set():
            status.update("platforms", p, state="done")
    print(json.dumps(status.data["platforms"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
