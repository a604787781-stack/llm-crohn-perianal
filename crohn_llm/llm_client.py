# -*- coding: utf-8 -*-
"""Model clients, JSON answer parsing and single-episode inference used in the study."""

from __future__ import annotations

import json
import os
import re
import time

from openai import OpenAI

from paths import DATA, ROOT
from prompt import build_messages

TRAIN_JSONL = DATA / "train_episodes.jsonl"
TEST_JSONL = DATA / "test_episodes.jsonl"
ALL_JSONL = DATA / "episode_inputs.jsonl"


def load_dotenv():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _fields(obj: dict, parse_error: str) -> dict:
    cd = obj.get("cd")
    prob = obj.get("probability")
    try:
        prob = float(prob) if prob is not None else None
    except (TypeError, ValueError):
        prob = None
    return {
        "cd": int(cd) if cd is not None else None,
        "probability": prob,
        "confidence": obj.get("confidence"),
        "findings": obj.get("findings"),
        "evidence": obj.get("evidence", []),
        "differential": obj.get("differential", ""),
        "parse_error": parse_error,
    }


def parse_json_answer(text: str) -> dict:
    empty = {"cd": None, "probability": None, "confidence": None, "findings": None,
             "evidence": [], "differential": ""}
    if not text:
        return {**empty, "parse_error": "empty"}
    t = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return _fields(json.loads(t), "")
    except Exception as e:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            try:
                return _fields(json.loads(m.group(0)), f"recovered:{type(e).__name__}")
            except Exception:
                pass
        cd = re.search(r'"cd"\s*:\s*"?([01])(?![\d.])', t)
        if cd:
            prob = re.search(r'"probability"\s*:\s*"?(\d+(?:\.\d+)?)', t)
            obj = {"cd": cd.group(1), "probability": prob.group(1) if prob else None}
            return {**_fields(obj, f"regex:{type(e).__name__}"), "raw": t[:500]}
        return {**empty, "parse_error": str(e), "raw": t[:500]}


def _load_pool(pool: str) -> list[dict]:
    if pool == "train":
        path = TRAIN_JSONL
    elif pool == "test":
        path = TEST_JSONL
    else:
        path = ALL_JSONL
    if not path.exists():
        raise SystemExit(f"缺少数据文件: {path}")
    return [json.loads(l) for l in path.open(encoding="utf-8")]


class DeepSeekBackend:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def chat(self, model, messages, temperature, thinking) -> tuple[str, dict]:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        )
        u = resp.usage
        return resp.choices[0].message.content or "", {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }


class OpenAICompatBackend:
    """硅基流动 / 中转站等 OpenAI 兼容接口。"""

    # 硅基流动上的混合思考模型默认开启思考，须显式关闭；其余模型不接受该字段
    HYBRID_THINKING = (
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-235B-A22B",
        "tencent/Hunyuan-A13B-Instruct",
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek-ai/DeepSeek-V3.1-Terminus",
        "deepseek-ai/DeepSeek-V3.2-Exp",
        "deepseek-ai/DeepSeek-V3.2",
        "qwen3-235b-a22b",
        "qwen3-14b",
        "qwen3-8b",
    )

    def __init__(
        self,
        api_key: str,
        base_url: str,
        reasoning_effort: str = "",
        thinking_budget: int = 0,
        max_tokens: int = 4096,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
        self.reasoning_effort = reasoning_effort
        # 思考无法关闭的模型（如 AntAngelMed）推理内容计入 max_tokens，需放宽
        self.max_tokens = max_tokens
        # 百川 M2/M3 无法关闭思考，只能限制预算（≥1024）
        self.thinking_budget = thinking_budget
        # 部分模型（如推理型 GPT）不接受 temperature / json 模式，首次报错后自动去掉并记住
        self.drop: set[str] = set()

    def chat(self, model, messages, temperature, thinking) -> tuple[str, dict]:
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        extra = {"enable_thinking": thinking} if model in self.HYBRID_THINKING else {}
        if self.thinking_budget:
            extra["thinking"] = {"budget_tokens": self.thinking_budget}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        for _ in range(3):
            call = {k: v for k, v in kwargs.items() if k not in self.drop}
            try:
                resp = self.client.chat.completions.create(**call, extra_body=extra)
                break
            except Exception as e:
                msg = str(e)
                bad = next(
                    (k for k in ("temperature", "response_format", "max_tokens")
                     if k in msg and k not in self.drop and "400" in msg),
                    None,
                )
                if not bad:
                    raise
                self.drop.add(bad)
        u = resp.usage
        det = getattr(u, "completion_tokens_details", None)
        msg = resp.choices[0].message
        reasoning_text = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get(
            "reasoning_content"
        )
        return msg.content or "", {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None,
            "reasoning_chars": len(reasoning_text or ""),
            "finish_reason": resp.choices[0].finish_reason,
        }


def run_one(
    backend,
    model: str,
    ep: dict,
    max_chars: int,
    thinking: bool,
    prompt_version: str = "v0_zero",
    temperature: float = 0.0,
) -> dict:
    text = ep["prompt_text"]
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n…（文本已截断）"
        truncated = True
    messages = build_messages(text, prompt_version)
    t0 = time.time()
    err = ""
    for attempt in range(4):
        try:
            content, usage = backend.chat(model, messages, temperature, thinking)
            parsed = parse_json_answer(content)
            err = ""
            break
        except Exception as e:
            content = ""
            usage = {}
            parsed = {"cd": None, "parse_error": "api"}
            err = f"{type(e).__name__}: {e}"
            transient = any(k in err for k in ("429", "RateLimit", "503", "502", "Timeout", "Connection"))
            if not transient:
                break
            time.sleep(5 * 2**attempt)
    sec = round(time.time() - t0, 2)
    gold = int(ep["cd_label"])
    pred = parsed.get("cd")
    return {
        "episode_id": ep["episode_id"],
        "patient_id": ep["patient_id"],
        "gold": gold,
        "pred": pred,
        "correct": int(pred == gold) if pred in (0, 1) else None,
        "prompt_version": prompt_version,
        "probability": parsed.get("probability"),
        "confidence": parsed.get("confidence"),
        "findings": parsed.get("findings"),
        "evidence": parsed.get("evidence"),
        "differential": parsed.get("differential"),
        "parse_error": parsed.get("parse_error", ""),
        "api_error": err,
        "truncated": truncated,
        "seconds": sec,
        "prompt_chars": len(text),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "ctx_overflow": usage.get("ctx_overflow", False),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "reasoning_chars": usage.get("reasoning_chars"),
            "finish_reason": usage.get("finish_reason"),
        },
        "raw_content": content,
    }


def summarize(results: list[dict], model: str, thinking: bool, meta: dict) -> dict:
    valid = [r for r in results if r["pred"] in (0, 1)]
    acc = sum(r["correct"] == 1 for r in valid) / len(valid) if valid else 0
    tp = sum(r["gold"] == 1 and r["pred"] == 1 for r in valid)
    tn = sum(r["gold"] == 0 and r["pred"] == 0 for r in valid)
    fp = sum(r["gold"] == 0 and r["pred"] == 1 for r in valid)
    fn = sum(r["gold"] == 1 and r["pred"] == 0 for r in valid)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return {
        "model": model,
        "thinking": thinking,
        **meta,
        "n": len(results),
        "n_valid": len(valid),
        "n_pos_gold": sum(r["gold"] == 1 for r in results),
        "n_neg_gold": sum(r["gold"] == 0 for r in results),
        "accuracy": round(acc, 4),
        "sensitivity": None if sens is None else round(sens, 4),
        "specificity": None if spec is None else round(spec, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_prompt_tokens": sum((r["usage"]["prompt_tokens"] or 0) for r in results),
        "total_completion_tokens": sum((r["usage"]["completion_tokens"] or 0) for r in results),
        "total_seconds": round(sum(r["seconds"] for r in results), 1),
    }
