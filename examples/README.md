# examples/

**All cases in this folder are synthetic.** They were written by the authors to mirror the structure and style of the inpatient records used in the study and do not correspond to any real patient. Names, record numbers and staff names are fictitious.

- `synthetic_raw_cases.json`: four admissions (two CD, two non-CD) before processing, with the history of present illness, past history, discharge status and colonoscopy, MR and CT report texts. They deliberately contain identifiers, sentences naming Crohn's disease and sentences mentioning IBD-specific drugs.
- `synthetic_episodes.jsonl`: the same admissions after `prepare_inputs.py` — identifiers replaced with `[已隐去]`, every sentence naming CD or an IBD-specific drug deleted, list numbering removed, and the six sections assembled. This is the exact input format sent to the models.

Regenerate the processed file with:

```bash
python crohn_llm/prepare_inputs.py examples/synthetic_raw_cases.json examples/synthetic_episodes.jsonl
```
