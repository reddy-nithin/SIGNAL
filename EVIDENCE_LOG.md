# SIGNAL Evidence Log

> Updated after each phase. Contains outputs, metrics, screenshots, and decisions.
> This file serves as: (1) knowledge base for the competition report, (2) demo preparation reference, (3) proof of work.

---

## Phase 0: Foundation
**Date started:** ___  |  **Date completed:** ___

### Dataset Audit Results
<!-- After downloading all datasets, fill in this table -->

| # | Dataset | Rows | Columns | Has Timestamps? | Has Subreddit Labels? | Has Substance Labels? | Notes |
|---|---|---|---|---|---|---|---|
| 1 | RMHD | | | | | | |
| 2 | Reddit MH Labeled | | | | | | |
| 3 | MH Reddit Cleaned | | | | | | |
| 4 | MH Cleaned Research | | | | | | |
| 5 | UCI Drug Review | | | | | | |
| 8 | DepressionEmo | | | | | | |

**Primary corpus decision:** [Which 2 datasets and why]

**Temporal strategy decision:** [Timestamps exist → temporal analysis OR No timestamps → cross-subreddit comparison]

### Environment Verification
- [ ] Vertex AI embedding test: [pass/fail + latency]
- [ ] Gemini test call: [pass/fail + latency]
- [ ] FAISS load knowledge chunks: [pass/fail + chunk count]
- [ ] FAERS JSON load: [pass/fail + signal count]
- [ ] `python -m signal.tests.test_setup` output:
```
[paste output here]
```

### Screenshots
<!-- Save screenshots to evidence/phase0/ and reference them here -->
<!-- Example: ![dataset audit](evidence/phase0/dataset_audit.png) -->

---

## Phase 1: Ingestion + Exemplars
**Date started:** ___  |  **Date completed:** ___

### Corpus Statistics
- Total posts ingested: ___
- Posts per source dataset: ___
- Embedding dimensions: ___
- FAISS index size: ___
- Embedding time: ___

### Stage Exemplar Curation
- Gemini pre-filtering: ___ posts processed → ___ high-confidence candidates
- Human validation: ___ accepted per stage

| Stage | Candidates Reviewed | Accepted | Researcher-Written | Total |
|---|---|---|---|---|
| Curiosity | | | | /50 |
| Experimentation | | | | /50 |
| Regular Use | | | | /50 |
| Dependence | | | | /50 |
| Crisis | | | | /50 |
| Recovery | | | | /50 |

- Stage centroid validation (nearest-neighbor sanity check):
```
Query: "I've been using again and can't stop"
Top 3 results: [paste]
```

### Screenshots
<!-- evidence/phase1/ -->

---

## Phase 2: Substance Resolution
**Date started:** ___  |  **Date completed:** ___

### Slang Lexicon
- Total entries: ___
- Drug classes covered: ___
- Sample resolutions tested:

| Input Slang | Rule-Based | Embedding | LLM | Ground Truth | All Correct? |
|---|---|---|---|---|---|
| "popping blues" | | | | counterfeit oxy/fentanyl | |
| "mixing bars and lean" | | | | alprazolam + codeine | |
| "been on subs for 6 months" | | | | buprenorphine (MAT) | |

### Method Comparison — Substance Detection

**Evaluation Source 1: UCI Drug Review ground truth**

| Method | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Rule-based | | | | |
| Embedding | | | | |
| LLM (Gemini) | | | | |
| Ensemble | | | | |

**Evaluation Source 2: Synthetic slang test set (50 cases)**

| Method | Correct Resolutions | Accuracy |
|---|---|---|
| Rule-based | /50 | |
| Embedding | /50 | |
| LLM (Gemini) | /50 | |

### Key Findings
<!-- What worked, what didn't, interesting disagreements between methods -->

### Screenshots
<!-- evidence/phase2/ -->

---

## Phase 3: Narrative Stage Classification
**Date started:** ___  |  **Date completed:** ___

### DistilBERT Training
- Training examples: ___ (after augmentation)
- Validation examples: ___
- Epochs: ___
- Best epoch: ___
- Training time: ___

**5-Fold Cross-Validation Results:**

| Fold | Macro F1 | Accuracy | Best/Worst Stage |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| **Mean ± Std** | | | |

**Training curve:**
<!-- Save loss/F1 curve plot to evidence/phase3/training_curve.png -->

### Method Comparison — Narrative Stage Classification

**Evaluation Layer A: Inter-method agreement (500 posts)**

| Method Pair | Cohen's Kappa | % Agreement |
|---|---|---|
| Rule-based vs DistilBERT | | |
| Rule-based vs Gemini | | |
| DistilBERT vs Gemini | | |
| Fleiss' Kappa (all 3) | | |

**Evaluation Layer B: Expert-annotated validation (100 posts)**

| Method | Macro F1 | Per-Stage F1 (Cur/Exp/Reg/Dep/Cri/Rec) | Accuracy |
|---|---|---|---|
| Rule-based | | | |
| DistilBERT | | | |
| LLM (Gemini) | | | |
| Ensemble | | | |

**Confusion Matrix (DistilBERT):**
```
[paste or reference image: evidence/phase3/confusion_matrix_distilbert.png]
```

**Confusion Matrix (Gemini):**
```
[paste or reference image: evidence/phase3/confusion_matrix_gemini.png]
```

### Key Findings
<!-- Which stages get confused? Where do methods disagree? What does disagreement reveal? -->
<!-- This section directly feeds the Results page of the competition report -->

### Screenshots
<!-- evidence/phase3/ -->

---

## Phase 4: Clinical Grounding + Multi-Substance KB
**Date started:** ___  |  **Date completed:** ___

### Knowledge Base Expansion
| Substance Class | Chunks Created | Source | Reviewed? |
|---|---|---|---|
| Opioids (existing) | 55 | opioid_track | ✅ |
| Alcohol | /10 | NIAAA/WHO + Gemini | |
| Benzodiazepines | /8 | FDA/literature + Gemini | |
| Stimulants | /8 | NIDA/literature + Gemini | |
| **Total** | /81 | | |

### FAISS Index Updated
- Total chunks indexed: ___
- Index build time: ___

### Retrieval Quality Spot-Check

| Query | Top Retrieved Chunk | Relevant? |
|---|---|---|
| "fentanyl overdose risk" | | |
| "alcohol withdrawal seizure" | | |
| "mixing xanax and opioids" | | |
| "methamphetamine neurotoxicity" | | |

### Pipeline End-to-End Test
```
Input: "I've been mixing lean with xans and I can't stop"

Layer 1 output (substances): [paste]
Layer 2 output (stage): [paste]
Layer 3 output (clinical context): [paste]
Layer 4 output (analyst brief): [paste]

Total latency: ___
```

### Screenshots
<!-- evidence/phase4/ -->

---

## Phase 5: Dashboard
**Date started:** ___  |  **Date completed:** ___

### Pages Completed
- [ ] Deep Analysis — paste text → full 4-layer report
- [ ] Narrative Pulse — stage distributions over time/subreddits
- [ ] Method Comparison — dual evaluation charts

### Demo Flow Test
- Demo example 1: [substance + stage + brief quality]
- Demo example 2: [substance + stage + brief quality]
- Demo example 3: [substance + stage + brief quality]
- Live analysis latency: ___
- Pre-cached examples working: [yes/no]

### Dashboard Screenshots
<!-- evidence/phase5/deep_analysis.png -->
<!-- evidence/phase5/narrative_pulse.png -->
<!-- evidence/phase5/method_comparison.png -->

---

## Phase 6: Polish + Submission
**Date started:** ___  |  **Date completed:** ___

### Competition Report
- [ ] Page 1: Problem & Approach (incl. related work)
- [ ] Page 2: Architecture & Methods
- [ ] Page 3: Results & Evaluation
- [ ] Page 4: Ethics & Impact
- [ ] Report saved to: [path]

### Demo Video
- [ ] 2-minute video recorded
- [ ] Video saved to: [path]

### Submission Checklist
- [ ] App deployed and accessible
- [ ] Report uploaded
- [ ] Demo video uploaded
- [ ] Submitted before noon April 6

---

## Metrics Summary (For Report Quick-Reference)

### Substance Detection
| Metric | Rule-Based | Embedding | LLM | Ensemble |
|---|---|---|---|---|
| Precision | | | | |
| Recall | | | | |
| F1 | | | | |

### Narrative Stage Classification
| Metric | Rule-Based | DistilBERT | LLM | Ensemble |
|---|---|---|---|---|
| Macro F1 | | | | |
| Accuracy | | | | |
| Kappa vs Expert | | | | |

### Key Numbers for Report
- Total posts in corpus: ___
- Knowledge chunks: ___
- FAERS signals: ___
- Slang lexicon entries: ___
- DistilBERT training examples: ___
- Expert-annotated validation posts: ___
- Pipeline latency: ___
- Substance classes covered: ___
