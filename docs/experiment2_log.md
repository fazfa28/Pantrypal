# Experiment 2 log: recipe recommendation

Working log for the report. Newest status first, then the history of decisions in order.
All numbers so far are from the **dev split (10 cases)**; nothing has been run on the test split yet.
Last updated: 2026-10-23.

## Current status

- **Frozen on 2026-10-23** after three dev tuning rounds; test run started the same day.
- The team chose to freeze **without** the planned improvements (stronger test cases, larger
  database); these are reported as limitations instead (see below).
- Last changes before the freeze, none of which affect what any arm outputs:
  - the diet check also flags Worcestershire sauce and bouillon (not vegetarian) and mayonnaise (not vegan);
  - useful@5 added to the judge summary, using the same per-recipe checks as the automatic metrics.

### Freeze record
SHA-256 prefixes of the frozen files. If any of these change, test results are no longer comparable.

| File | SHA-256 (first 12) |
|---|---|
| `eval/run_recommend.py` | c6d34e54c999 |
| `eval/judge_recommend.py` | 29c67e37860b |
| `core/recipes.py` | 36d2dade34a5 |
| `core/assistant.py` | c770fc29ff16 |
| `core/canon.py` | 44ae547e9289 |
| `eval/cases/exp2_cases.jsonl` | 9c74aeea398c |
| `data/recipes_5k.parquet` | 5577baddefa5 |
| `data/aliases_extra.yaml` | 60b7d11ca912 |
| `config.yaml` | ae1877afeb52 |

## Research question

Does PantryPal's design (an LLM to understand the request, code to retrieve and check recipes)
recommend better recipes than the usual alternatives: asking a chatbot directly (LLM-only), or
standard retrieval-augmented generation (RAG)?

## Test results (50 cases, run 2026-10-23)

Full table with CIs and paired tests: `eval/results/exp2_report_test.md` (`python -m eval.exp2_stats --split test`).
Report: https://claude.ai/artifact/Ka52NXSnSRRNL2tmHx5tzc

| Test, 50 cases | LLM-only | RAG | PantryPal |
|---|---|---|---|
| Grounded | 0.02 | 1.00 | 1.00 |
| Cookable@5 | 0.84 (self-reported) | 0.15 | 0.78 |
| Diet / time / exclusion respected | 0.96 / 1.00 / 1.00 | 0.80 / 0.88 / 0.89 | 1.00 / 1.00 / 1.00 |
| Use-up | 0.96 | 0.60 | 0.69 |
| Hallucinated pantry items | 6% | – | – |
| Filters correct | – | – | 0.98 |
| Preference fit (judge) | 4.67 | 3.48 | 3.49 |
| Sensible | 0.99 | 0.97 | 0.92 |
| useful@5 | 0.76 (self-reported) | 0.09 | 0.37 |
| Cases with at least one useful recipe | 46/50 | 13/50 | 34/50 |

- PantryPal vs RAG (Wilcoxon): better on cookable, diet and useful@5 (all p ≤ 0.001); same fit (p = 0.98).
- PantryPal vs LLM-only: lower fit and useful@5 (p < 0.001); cookable and diet not significantly different.
- 33/50 cases had fewer than 20 cookable recipes; all 55 non-cookable PantryPal picks came from the top-up.
- Second judge (Qwen3-VL 235B, 10 cases, 146 recipes): Spearman 0.50, weighted kappa 0.50 with GPT-6.1 Sol; same ranking of arms.
- Incident: GPT-6.1 Sol hit OpenRouter's new-account limit (20 requests/min); the judge was re-run with 1 worker
  (resumed; no ratings lost or changed). Frozen file fingerprints were checked unchanged after the run.
- Cost of the test split: arms US$0.05, judges US$0.32.

## Setup

### Recipe database
- Food.com Recipes and Reviews (Kaggle), filtered to 5,000 recipes: at most 12 ingredients, at most
  60 minutes, at least 1 review, most-reviewed first.
- Recipes must have **at least 3 non-staple ingredients** (drops near-empty "salt, oil, egg" recipes).
- Recipes whose ingredient-name list is shorter than their quantity list are **dropped**: Food.com
  silently loses some ingredient names (e.g. a "Chicken Noodle Soup" with no chicken).
- Staples (always assumed available, ignored in matching): salt, black pepper, oil, water, sugar.
- Each recipe keeps its Food.com category, used for the meal-type rule (dinner / breakfast / dessert).

### Ingredient canonicalisation (prerequisite)
Every recipe ingredient and pantry item is mapped to a canonical food id.
- **Bug found:** the old fuzzy matcher (`token_set_ratio`) scored "fish sauce" as a perfect match for
  "fish", "baking soda" for "soda", "beef broth" for "beef", etc. The canonical list had only 70 ids.
- **Fix:** an LLM (GLM 5.3 Flash) mapped every distinct ingredient string once, offline, into a lookup
  table (`data/aliases_extra.yaml`, about 400 ids now); the fuzzy fallback became typo-only (`fuzz.ratio`).
  One error fixed by hand ("ground ginger" was mapped to fresh ginger).
- **Measured on a fixed random sample of 100 ingredient strings** (`eval/mapping_check/`):

  | | Before | After |
  |---|---|---|
  | Wrong match | 19% | 1% |
  | Precision (correct / matched) | 74% | 99% |
  | Coverage (matched) | 73% | 100% |

- Caveats: the after-check is in-sample (the table was built from these strings); labels were made by
  the AI and still need a teammate's review.

### Test cases (`eval/make_cases.py`, `eval/cases/exp2_cases.jsonl`)
- 60 cases, 10 dev and 50 test, fixed seed.
- Each case = **pantry** + **natural-language request**, generated from a structured **spec**
  that is the ground truth for the automatic checks:
  diet (vegetarian / vegan), time limit (15–45 min), an ingredient to exclude, an ingredient to
  use up, a soft preference (12 types, e.g. "something spicy", "an Italian-style dish"), servings.
  Every case tests at least two of these.
- **Pantry** = the non-staple ingredients of 2–3 database "anchor" recipes that satisfy the hard
  constraints (so a good answer exists) + 3–5 common extras. Anchors are not the correct answers
  and are never shown to any arm.
- Use-up items are perishable (produce, dairy, meat, ...); excluded items are placed in the pantry.
- Request example: "What can I make for dinner for 2 people? It must be vegan. I don't like
  lettuce. The green bean needs using soon."

### Arms (`eval/run_recommend.py`)
All arms use the same chat model, **GLM 5.3 Flash**, temperature 0. Each returns 5 recipes.

| Arm | Retrieval | Final step |
|---|---|---|
| LLM-only | none | the model writes 5 recipes (name, minutes, ingredients, pantry items used) |
| RAG | MiniLM embedding of "request + I have: pantry"; 20 most similar recipes; **no filters** | the model picks 5 by id |
| PantryPal | the model turns the request into filters (forced `search_recipes` tool call); code applies them, keeps only recipes **cookable** from the pantry, ranks those by similarity to the request (same MiniLM embeddings), returns 20 | the model picks 5 by id, **same prompt as RAG** |

RAG and PantryPal share the selection prompt and candidate count, so only retrieval differs.
The PantryPal retrieval lives in `core/recipes.py` (`search(..., query=...)`), so the app uses the same method.

### Metrics
Per case, each metric is a share of the 5 slots (a missing pick counts as a failure), then averaged over cases.

| Metric | How it is computed |
|---|---|
| Grounded | Recipe exists in the database. RAG/PantryPal must pick by id from the offered candidates (others are dropped). LLM-only counts only if its name exactly matches a database recipe. |
| Cookable@5 | At most 1 non-staple ingredient missing **and** at least 3 in stock. Same rule for all arms, no substitute credit. LLM-only uses its own (self-reported) ingredient list. |
| Diet respected | No ingredient in a forbidden category (meat/seafood; + dairy/egg for vegan) and no animal word in the ingredient text or recipe name (plant-based phrases like "soy milk" removed first). Only cases with a diet. |
| Time respected | Minutes at most the limit (database time; LLM-only's claimed time). Only cases with a limit. |
| Exclusion respected | Excluded id not in the recipe and its name not in any ingredient. Only cases with an exclusion. |
| Use-up | Recipe contains the use-up item. |
| Hallucinated (LLM-only) | Share of pantry items it claims to use that are not in the pantry. |
| Filters correct (PantryPal) | The extracted filters (time, diet, meal, exclusion) exactly match the spec. |
| Preference fit (judge) | 1–5 against a rubric, by GPT-6.1 Sol. One call per case; recipes from all arms pooled, de-duplicated and shuffled, so the judge is blind to the arm. Diet/time/exclusion/pantry are explicitly excluded from fit. |
| Sensible (judge) | Yes/no: a coherent dish the listed ingredients can make. |
| useful@5 | Cookable **and** every hard constraint respected **and** judge fit at least 4. Also: number of cases with at least one useful recipe. (Computed ad hoc so far; to be added to the scripts.) |
| Cost, latency | From saved token counts and config prices. |

Judges: GPT-6.1 Sol (primary, all cases); Qwen3-VL 235B (secondary, about 20% of cases) for
judge–judge agreement; 2–3 teammates rate 40 blind items for human–judge agreement.

## Dev results

### Latest (after round 3)

| Dev, 10 cases | LLM-only | RAG | PantryPal |
|---|---|---|---|
| Grounded | 0.00 | 1.00 | 1.00 |
| Cookable@5 | 0.92 (self-reported) | 0.16 | 0.90 |
| Diet respected | 0.97 | 0.80 | 1.00 |
| Time respected | 1.00 | 1.00 | 1.00 |
| Exclusion respected | 1.00 | 1.00 | 1.00 |
| Use-up | 0.93 | 0.67 | 0.63 |
| Hallucinated pantry items | 8% | n/a | n/a |
| Filters correct | n/a | n/a | 1.00 |
| Preference fit (judge, 1–5) | 4.56 | 4.06 | 3.78 |
| Sensible (judge) | 1.00 | 1.00 | 0.96 |
| useful@5 | 0.80 | 0.12 | 0.56 |
| Cases with at least one useful recipe | 10/10 | 5/10 | 10/10 |
| US$ per request (arm) | 0.00036 | 0.00048 | 0.00044 |

Reading: against RAG (both grounded), PantryPal is far better on cookability, constraints and
useful@5. LLM-only scores highest on fit, but its recipes are invented and unverifiable
(no real recipe, self-reported ingredients and times, 8% hallucinated pantry items).
PantryPal trades some preference fit for real, code-checked recipes.

### Tuning history (dev only)

| Round | Change | PantryPal fit | Why |
|---|---|---|---|
| 1 | Baseline: filters, then rank by coverage score | 3.42 | Coverage favours tiny side dishes ("Paprika Potatoes" is 100% covered), so the 20 candidates were mostly potato and rice sides |
| 2 | Feasibility gate, then rank by similarity to the request; prompt asks to set `meal` | 3.60 | Better candidates, but the model still often left the meal/time filters empty |
| 3 | `search_recipes` filters made **required** (with null / "none" / "any") | 3.78 | Filter extraction went to 10/10 correct (it had missed meal, time limit and exclusions in several cases) |

Earlier fixes found during dev: the diet check flagged "soy milk" and "peanut butter" as animal
products (fixed with a plant-phrase list); API calls could hang (2-minute timeout added).
Archived outputs of earlier rounds: `eval/results/archive/v1_*`, `v2_*`.

## Earlier design (superseded)

The first version compared PantryPal's match score with a Jaccard-overlap baseline and LLM-only on
random inventories, measuring only cookable@5. Results on 50 random inventories:
- With the original "at most 1 missing" rule: PantryPal 1.00, Jaccard 0.20. This was an artefact:
  PantryPal picked recipes with only 1–2 non-staple ingredients, which count as cookable by default.
- With the fixed rule (at least 3 in stock): PantryPal 0.27, Jaccard 0.14.
- No-staples ablation (old rule): PantryPal 0.97, Jaccard 0.10.

It was replaced because it did not test preferences, which is the reason PantryPal uses an LLM;
random inventories were unrealistic; Jaccard was a weak baseline. The old runner
(`eval/run_retrieval.py`) remains as a free ablation of the ranking formula.

## Known limitations and validity checks

- **Constraint checks are rules, not ground truth.** Known misses: Worcestershire sauce (anchovy),
  mayonnaise (egg, for vegan), hidden forms of an excluded item (e.g. "no egg" vs egg pasta).
  Known false positive: "hamburger buns". To do: validate on about 60 hand-labelled
  recipe–constraint pairs and report precision and recall.
- **Sensitivity of the checks** (would they catch a system that ignores the constraint?).
  If PantryPal's ranking ignores a constraint, the share of top-5 picks that violate it:
  time 40% (36/44 cases with at least one), diet 14% (14/34), exclusion 10% (6/23).
  Diet and exclusion are too easy: only 15/34 diet cases have tempting items in the pantry, and
  excluded items are rarely used by cookable recipes. To do: strengthen the test cases.
- **LLM-only times and ingredient lists are self-reported** and cannot be verified.
- **Judge fit ignores pantry coverage** by design; useful@5 combines both.
- **Use of dev:** three tuning rounds so far; dev numbers are optimistic. Test numbers are the result.
- **Database size:** with 5,000 recipes, many pantries can only make simple sides, which caps
  PantryPal's preference fit.

## Next steps

1. ~~Add useful@5 to the scripts~~ (done) and ~~freeze~~ (done, 2026-10-23).
2. ~~Run all arms and both judges on the 50 test cases~~ (done; results above). ~~Statistics~~ (done, `eval/exp2_stats.py`).
3. Human checks: 40 blind items rated by 2–3 teammates (judge agreement); about 60 recipe–constraint
   pairs labelled (checker precision and recall).
4. Statistics: paired comparisons of PantryPal vs each baseline (bootstrap CIs, Wilcoxon for fit,
   McNemar/sign tests for yes/no outcomes); write-up.

Not done, reported as limitations (possible future work): stronger diet/exclusion temptations in the
test cases; a larger recipe database (about 20,000 recipes).

## Cost so far (approximate)

| Item | US$ |
|---|---|
| Ingredient mapping (about 2,200 strings) | < 0.05 |
| Dev runs, all rounds | < 0.05 |
| Judge on dev (3 rounds) | 0.19 |
| **Total** | **about 0.30** |

## Files

| File | What |
|---|---|
| `eval/make_cases.py` | Builds the 60 test cases and the review sheet |
| `eval/run_recommend.py` | Runs the three arms; automatic metrics |
| `eval/judge_recommend.py` | LLM judge; blind human sheet; agreement statistics |
| `core/recipes.py` | Index build, filters, feasibility gate, semantic ranking, diet and meal rules |
| `core/assistant.py` | `search_recipes` tool (shared with the app) |
| `scripts/build_ingredient_map.py`, `scripts/sample_mappings.py` | Ingredient mapping and its before/after check |
| `eval/results/exp2__<arm>__<split>.jsonl` | Per-case outputs of each arm |
| `eval/results/exp2_judge__<judge>__<split>.jsonl` | Judge ratings |
