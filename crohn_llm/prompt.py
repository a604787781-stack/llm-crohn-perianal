# -*- coding: utf-8 -*-
"""任务 A：克罗恩病表型识别 — 多版本 Prompt。

版本（逐级叠加，便于消融）：
  v0_zero      零样本基线
  v1_criteria  v0 + 诊断要点与鉴别诊断知识
  v2_fewshot   v0 + 训练集参考病例（少样本）
  v3_cot       v0 + 先结构化提取证据、再下结论
  v4_full      诊断要点 + 参考病例 + 结构化推理
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from paths import DATA

# Few-shot exemplars are real admissions and are therefore not distributed.
FEWSHOT_FILE = DATA / "fewshot_examples.json"
TRAIN_JSONL = DATA / "train_episodes.jsonl"

PROMPT_VERSIONS = ["v0_zero", "v1_criteria", "v2_fewshot", "v3_cot", "v4_full"]

ROLE = """你是消化病学临床专家，任务是根据脱敏后的住院病历文本，判断该患者本次住院是否为克罗恩病（Crohn's disease, CD）。"""

BACKGROUND = """背景说明：
1. 这是回顾性病历表型识别：材料包括本次入院现病史、既往史、出院情况，以及出院前的肠镜、MR、CT 报告。
2. 文本已去除个人信息（显示为 [已隐去]）；含诊断名称的语句已被整句删除，因此行文可能不连贯，请勿把「内容缺失」本身当作诊断依据。
3. 只根据给定文本判断，不要臆造未提及的检查结果。信息不足时仍须给出 0 或 1，并相应调整 probability。"""

CRITERIA = """诊断参考要点（依据炎症性肠病诊治共识的常用标准概括，仅供推理参考）：
支持克罗恩病：
- 临床：青中年多见；慢性反复腹痛、腹泻、体重下降、营养不良；可伴发热、贫血；肠外表现（关节、皮肤、眼、口腔溃疡）。
- 肛周病变：肛瘘（尤其复杂性、多发、高位或反复发作）、肛周脓肿、肛管溃疡、水肿性皮赘；肛周病变可为首发表现。
- 内镜：病变节段性/跳跃性分布；纵行溃疡、阿弗他溃疡、鹅卵石样外观；好发于回肠末端及右半结肠；肠腔狭窄；直肠常相对不受累。
- 影像（CTE/MRE/盆腔 MR）：肠壁节段性增厚伴分层强化（靶征）、梳状征、系膜脂肪增生；狭窄伴近端扩张；内瘘、窦道、腹腔或盆腔脓肿；复杂肛瘘伴直肠壁增厚强化。
- 病程：慢性、复发—缓解交替，或既往有肠切除、肠梗阻、肠瘘史。
需要鉴别（支持非克罗恩病）：
- 单纯性（腺源性）肛瘘/肛周脓肿：病变局限于肛周，无肠道炎症证据，无慢性腹部症状。
- 溃疡性结肠炎：自直肠起始的连续性、弥漫性浅表炎症，一般无小肠受累和肛瘘。
- 肠结核：环形溃疡、回盲瓣固定开放、结核感染证据或肺结核病史。
- 肠白塞病：孤立的回盲部大溃疡，伴反复口腔、生殖器溃疡。
- 其他：淋巴瘤及肠道肿瘤、缺血性肠炎、感染性肠炎、放射性肠炎、药物性肠损伤等。"""

COT = """推理要求：请先逐项从文本中提取证据，写入 "findings"，再综合判断。每项只写文本中明确提到的内容，未提及写「未提及」：
- symptoms：消化道症状及病程
- perianal：肛周病变情况
- endoscopy：肠镜表现（部位、分布、溃疡形态、狭窄）
- imaging：CT/MR 表现（肠壁、狭窄、瘘管、脓肿、系膜）
- history：既往史中的相关疾病、手术与治疗
- against：不支持克罗恩病或提示其他疾病的依据"""

OUTPUT_BASE = """输出要求：必须严格输出一个 JSON 对象，不要输出 Markdown 代码块或其他文字。
JSON 字段：
{findings_line}- "cd": 0 或 1（1=克罗恩病，0=非克罗恩病），须与 probability 一致（probability ≥ 50 时为 1）
- "probability": 0~100 的整数，表示该患者为克罗恩病的概率
- "evidence": 字符串数组，1~5 条最关键的判断依据（简短中文）
- "differential": 字符串，最需要鉴别的其他疾病（可为空字符串）"""

FINDINGS_LINE = """- "findings": 对象，包含 symptoms、perianal、endoscopy、imaging、history、against 六个字符串字段（须放在最前）
"""

USER_TEMPLATE = """请根据下列病历材料判断是否为克罗恩病，并按要求输出 JSON。

{prompt_text}
"""

FEWSHOT_HEADER = """以下是若干已明确最终诊断的参考病例（来自同一数据来源，同样经过脱敏与病名语句删除），用于帮助你理解材料格式和判断尺度："""

FEWSHOT_MAX_CHARS = 2500


def _uses(version: str) -> tuple[bool, bool, bool]:
    """返回 (criteria, fewshot, cot)。"""
    table = {
        "v0_zero": (False, False, False),
        "v1_criteria": (True, False, False),
        "v2_fewshot": (False, True, False),
        "v3_cot": (False, False, True),
        "v4_full": (True, True, True),
    }
    if version not in table:
        raise ValueError(f"未知 prompt 版本: {version}，可选 {PROMPT_VERSIONS}")
    return table[version]


def build_fewshot_examples(n_pos: int = 2, n_neg: int = 2, seed: int = 2026) -> list[dict]:
    """从训练集抽取参考病例（每患者 1 条，长度适中），固定写入文件以保证可复现。"""
    rows = [json.loads(l) for l in TRAIN_JSONL.open(encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["patient_id"]].append(r)
    rng = random.Random(seed)

    def pool(label: int) -> list[dict]:
        cands = []
        for pid, items in by.items():
            labels = {int(x["cd_label"]) for x in items}
            if labels != {label}:
                continue
            ok = [x for x in items if 1200 <= len(x["prompt_text"]) <= FEWSHOT_MAX_CHARS]
            if ok:
                cands.append(rng.choice(ok))
        rng.shuffle(cands)
        return cands

    picked = pool(1)[:n_pos] + pool(0)[:n_neg]
    rng.shuffle(picked)
    examples = [
        {
            "episode_id": x["episode_id"],
            "patient_id": x["patient_id"],
            "cd_label": int(x["cd_label"]),
            "prompt_text": x["prompt_text"],
        }
        for x in picked
    ]
    FEWSHOT_FILE.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    return examples


def load_fewshot_examples() -> list[dict]:
    if not FEWSHOT_FILE.exists():
        return build_fewshot_examples()
    return json.loads(FEWSHOT_FILE.read_text(encoding="utf-8"))


def fewshot_patient_ids() -> set[str]:
    """参考病例所属患者，评估时应剔除。"""
    return {x["patient_id"] for x in load_fewshot_examples()}


def _fewshot_block() -> str:
    parts = [FEWSHOT_HEADER]
    for i, ex in enumerate(load_fewshot_examples(), 1):
        dx = "克罗恩病（cd=1）" if ex["cd_label"] == 1 else "非克罗恩病（cd=0）"
        parts.append(f"—— 参考病例 {i}｜最终诊断：{dx} ——\n{ex['prompt_text'].strip()}")
    parts.append("—— 参考病例结束 ——")
    return "\n\n".join(parts)


def build_system_prompt(version: str = "v0_zero") -> str:
    criteria, _, cot = _uses(version)
    blocks = [ROLE, BACKGROUND]
    if criteria:
        blocks.append(CRITERIA)
    if cot:
        blocks.append(COT)
    blocks.append(OUTPUT_BASE.format(findings_line=FINDINGS_LINE if cot else ""))
    return "\n\n".join(blocks)


def build_messages(prompt_text: str, version: str = "v0_zero") -> list[dict]:
    _, fewshot, _ = _uses(version)
    user = USER_TEMPLATE.format(prompt_text=prompt_text)
    if fewshot:
        user = _fewshot_block() + "\n\n=== 待判断病例 ===\n\n" + user
    return [
        {"role": "system", "content": build_system_prompt(version)},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    import sys

    v = sys.argv[1] if len(sys.argv) > 1 else "v0_zero"
    for m in build_messages("（病历正文）", v):
        print(f"##### {m['role']}\n{m['content']}\n")
