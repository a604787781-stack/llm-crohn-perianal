# data/

Patient-level data are not distributed. The medical records used in the study are available from the corresponding author on reasonable request, subject to approval by the institutional ethics committee and a data-use agreement. Everything in this folder except this file is excluded by `.gitignore`.

Files expected by the scripts:

| File | Produced by | Content |
|---|---|---|
| `episode_inputs.jsonl` | `prepare_inputs.py` | One admission per line (format below) |
| `train_episodes.jsonl`, `test_episodes.jsonl` | `split_dataset.py` | Development and held-out test sets |
| `split.json`, `split_patients.csv` | `split_dataset.py` | Patient-level split |
| `fewshot_examples.json` | `prompt.py` (first use) | The four few-shot exemplars (two CD, two non-CD) drawn from the development set |
| `pilot_episodes.json` | user-supplied | Admissions used during prompt development (JSON list with `patient_id`); used only for one sensitivity analysis |

Format of one admission:

```json
{
  "episode_id": "P001__1",
  "patient_id": "P001",
  "cd_label": 1,
  "prompt_text": "【现病史】\n…\n\n【既往史】\n…\n\n【出院情况】\n…\n\n【肠镜报告】\n…\n\n【MR报告】\n…\n\n【CT/CTE报告】\n…",
  "meta": {"has_colonoscopy": 1, "has_mr": 1, "has_ct": 1, "is_first_admission": 1, "...": "..."}
}
```

See `examples/synthetic_episodes.jsonl` for complete (synthetic) examples.
