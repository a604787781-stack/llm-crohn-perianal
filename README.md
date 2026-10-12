# Large language models identify Crohn's disease from inpatient records of patients with perianal fistula

Core code for the study evaluating eight large language models (LLMs), each with five prompting strategies, for identifying Crohn's disease (CD) from de-identified Chinese inpatient records of patients with perianal fistula or abscess, after every sentence naming CD or an IBD-specific drug had been deleted.

> Xue F, Xiao G, Hu X, Lei M, Ma Y, Lin H. Large language models identify Crohn's disease from inpatient records of patients with perianal fistula. *npj Digital Medicine* (under review).

This repository contains the core pipeline. Record retrieval, text extraction from the hospital's document archive and figure generation are institution-specific and are not included. **No patient data are included**; the examples are synthetic.

## Contents

| File | Purpose | Manuscript section |
|---|---|---|
| `crohn_llm/deidentify.py` | Replacement of personal identifiers; deletion of every sentence naming CD or an IBD-specific drug; removal of list numbering and deletion artefacts | Methods – Construction of model input |
| `crohn_llm/prepare_inputs.py` | Assembly of the six labelled input sections from raw record text | Methods – Construction of model input |
| `crohn_llm/split_dataset.py` | Patient-level split stratified by CD status (80%/20%, seed 42) | Methods – Participants and admission assembly |
| `crohn_llm/prompt.py` | The five prompting strategies (zero-shot, criteria, few-shot, chain-of-thought, combined) | Methods – Prompting strategies; Supplementary Note 1 |
| `crohn_llm/llm_client.py` | API clients, JSON answer parsing, retries and single-admission inference | Methods – Large language models |
| `crohn_llm/run_experiment.py` | Runs the 8 models × 5 prompts with checkpointing and failure re-runs | Methods – Configuration selection and test evaluation |
| `crohn_llm/keyword_baseline.py` | Keyword rules A and B | Methods – Keyword baseline |
| `crohn_llm/select_config.py` | Pre-specified configuration selection on the development set | Methods – Configuration selection and test evaluation |
| `crohn_llm/analyze.py` | Test-set analysis: patient-clustered bootstrap, DeLong test, Holm correction, GEE, calibration, stability, sensitivity and subgroup analyses | Methods – Statistics and reproducibility |
| `crohn_llm/demo.py` | Runs the synthetic examples through one model and prompt (or prints the prompt with `--dry-run`) | — |
| `examples/` | Four synthetic admissions, before and after processing | — |

Code comments and prompts are in Chinese, the language of the clinical records; an English translation of the prompts is given in Supplementary Note 1 of the paper.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then fill in the API keys you need
```

The study used Python 3.13.

## Quick start with the synthetic examples

```bash
# 1. De-identify and assemble the model input
python crohn_llm/prepare_inputs.py examples/synthetic_raw_cases.json examples/synthetic_episodes.jsonl

# 2. Inspect the exact messages sent to the model (no API call)
python crohn_llm/demo.py --dry-run --prompt v1_criteria

# 3. Call a model (requires the corresponding API key in .env)
python crohn_llm/demo.py --model dsflash --prompt v0_zero
```

The few-shot and combined prompts (`v2_fewshot`, `v4_full`) contain four real admissions from the development set and therefore require `data/fewshot_examples.json`, which is not distributed.

## Full pipeline (with access to the data)

```bash
python crohn_llm/prepare_inputs.py <raw_cases.json> data/episode_inputs.jsonl
python crohn_llm/split_dataset.py                       # data/train_episodes.jsonl, data/test_episodes.jsonl
python crohn_llm/keyword_baseline.py                    # keyword rules on the development set
python crohn_llm/run_experiment.py --split train        # 40 configurations on the development evaluation set
python crohn_llm/select_config.py                       # pre-specified selection rule
python crohn_llm/run_experiment.py --split test         # all configurations, once, on the held-out test set
python crohn_llm/run_experiment.py --split test --models gpt55 antangelmed --prompts v0_zero --repeat-tag _rep2
python crohn_llm/run_experiment.py --split test --models gpt55 antangelmed --prompts v0_zero --repeat-tag _rep3
python crohn_llm/analyze.py                             # outputs/formal/test_analysis.json and .md
```

The expected data format is described in `data/README.md`. Model identifiers, inference settings (temperature 0, output limits, reasoning disabled where possible) and platform endpoints are defined in `crohn_llm/run_experiment.py`. Commercial models are updated frequently, so re-running the code may not reproduce the published outputs exactly.

## Data availability

The medical records used in the study are not publicly available. De-identified model inputs, reference labels and model outputs are available from the corresponding author on reasonable request, subject to approval by the institutional ethics committee and a data-use agreement.

## Contact

Hongcheng Lin, Department of Coloproctology, The Sixth Affiliated Hospital, Sun Yat-sen University (lhcheng@mail.sysu.edu.cn).

## License

MIT (see `LICENSE`).
