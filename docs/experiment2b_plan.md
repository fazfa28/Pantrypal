# Experiment 2b: follow-up to Experiment 2 (plan, written before any run)

Written 2026-10-25, after the Experiment 2 test results were known. This is a **post-hoc follow-up**,
designed to fix weaknesses in Experiment 2's evaluation; it is reported as such. Experiment 2's frozen
files and results are not changed. Everything below is fixed before the first run; no tuning on test.

## Why
1. Plain RAG ignored the pantry, so it was a weak baseline.
2. LLM-only's cookability was self-reported (its own ingredient lists), and "real recipe" only meant
   "in our 5,000-recipe database", which is not a quality measure.
3. We did not know which parts of PantryPal matter.

## Methods (GLM 5.3 Flash, temperature 0, same 50 test cases)
| Method | Definition | New? |
|---|---|---|
| llm_only | Experiment 2 output, reused | no |
| rag | Experiment 2 output, reused (reference only) | no |
| rag_pantry | Same query as rag ("request I have: pantry"); 100 most similar recipes (MiniLM); re-ranked by pantry coverage (share of the recipe's non-staple ids in the pantry; ties by similarity); top 20; the model picks 5 with the Experiment 2 selection prompt. No filters. | yes |
| pantrypal | Experiment 2 output, reused | no |
| pantrypal_noai | Experiment 2 PantryPal candidates (same filters, same order); the top 5 are taken directly, no model choice. No API call. | yes |
| pantrypal_labels | Experiment 2 PantryPal candidates; each candidate also shows `missing_from_pantry`; the selection prompt adds "(each candidate lists what is missing_from_pantry; prefer those with nothing or little missing)". | yes |

## Steps for LLM-only recipes
Generated on request by GLM 5.3 Flash with the app's `write_steps` prompt (core/recommend.py), so that
the judge sees name, ingredients and steps for every recipe. Database recipes use their Food.com steps,
capped at the first 12 steps for judging.

## Measures
| Measure | By | Notes |
|---|---|---|
| Source | description | database or AI-generated; no longer scored as "real" |
| Cookable | code, Experiment 2 rule (≤ 1 non-staple id missing, ≥ 3 in pantry) | LLM-only: its ingredient list **plus** the essential ingredients GPT-6.1 Sol says are missing (completeness audit, name + ingredients only) |
| Diet / time / avoid / use-up | code, Experiment 2 checks | Diet and avoid double-checked by GPT-6.1 Sol; agreement reported |
| Fit (1–5) | GPT-6.1 Sol | Experiment 2 rubric; now sees steps too |
| Workable (1–5) | GPT-6.1 Sol | "Would this recipe work as written?"; told to ignore writing quality |

All six methods' recipes for a case are pooled, de-duplicated and shuffled in one judge call (blind).
Second judge: Qwen3-VL 235B on 10 cases (fixed seed), fit and workable.
Per case, each measure is a share or mean over the 5 slots (a missing pick counts as a failure; fit and
workable = 1). PantryPal is compared with every other method by paired Wilcoxon tests over cases;
95% bootstrap CIs over cases. No combined score (useful@5) is headlined.

## Known limits (stated in advance)
- The judge may favour AI-written text (tidy steps); it is told to judge whether a recipe works, not its writing.
- Steps for LLM-only recipes were written after the fact; they were not part of its original answer.
- No humans: all opinion measures come from AI judges.

Budget: about US$2. Code: `eval/exp2b.py`.
