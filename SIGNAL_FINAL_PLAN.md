# SIGNAL: Substance Intelligence through Grounded Narrative Analysis of Language

## Context

**Competition:** NSF NRT Project Challenge at UMKC 2026 Spring Research-A-Thon (Challenge 1: AI for Substance Abuse Risk Detection from Social Signals)
**Deadline:** April 6, 2026 (submission) | April 10, 2026 (live demo)
**Developer:** Solo
**Track:** Track A — AI Modeling and Reasoning
**Evaluation:** Technical quality (40%), Innovation (30%), Impact/relevance (20%), Clarity (10%)

---

## The Core Thesis

Every other team will answer: **"Is this post risky?"**

SIGNAL answers: **"What chapter of the addiction story is this person living — and what does clinical science say about what comes next?"**

Substance abuse follows recognizable narrative arcs documented in addiction medicine (Prochaska's Transtheoretical Model, Jellinek Curve, NIDA's Cycle of Addiction). SIGNAL operationalizes these arcs as a classification task on social media text, then grounds every detection in real pharmacological data that no other team possesses.

The result: for any social media post, SIGNAL produces a multi-layered clinical intelligence report — substance resolution, narrative stage classification, pharmacological risk context, and an evidence-cited analyst brief. At the population level, tracking narrative stage distributions over time reveals community escalation patterns without any complex temporal modeling libraries.

---

## Why SIGNAL Wins

| What judges will see from other teams | What judges will see from SIGNAL |
|---|---|
| "This post mentions opioids" → risk label | "This post describes fentanyl-contaminated counterfeit pills during an active dependence narrative — FAERS data shows respiratory depression signal (PRR 4.2) for this combination" |
| One classification task (risky vs not) | Two classification dimensions (substance detection + narrative stage), each with 3-method comparison |
| Generic LLM summary | Evidence-cited analyst brief grounded in 204 FAERS signals + 55 pharmacological knowledge chunks |
| Static risk label | Narrative trajectory — "this community shifted from 60% experimentation posts to 45% dependence posts in 8 weeks" |
| No clinical grounding | Every substance resolved to clinical entity with receptor binding, LD50, and adverse event data |

**The unfair advantages (confirmed working):**
- 55 curated opioid pharmacological knowledge chunks → FAISS/BM25 indexed and retrievable
- 204 FAERS consensus signals → loadable JSON with PRR/ROR/EBGM scores
- Working FAISS + BM25 hybrid retrieval pipeline
- Working Vertex AI / Gemini integration
- Multi-substance knowledge base (opioids + alcohol + benzodiazepines + stimulants — ~81 chunks total; see Phase 4)

---

## Theoretical Grounding: Narrative Stages

SIGNAL's narrative stage classification draws from established addiction science:

| Stage | Description | Textual Signals | Clinical Implication |
|---|---|---|---|
| **Curiosity** | Pre-use interest, questions, peer observation | "What does X feel like?", "My friend tried...", "Is it safe to..." | Prevention window — intervention most effective here |
| **Experimentation** | First/early use, recreational framing | "Tried it last weekend", "Not addicted, just fun", dosage questions | Early education opportunity |
| **Regular Use** | Patterned use, functional framing | "I use every Friday", "helps me deal with...", tolerance mentions | Escalation risk — tolerance signals dependence pathway |
| **Dependence** | Compulsive use, withdrawal, loss of control | "Can't function without", "sick when I stop", "need more to feel it" | Clinical intervention critical — withdrawal management needed |
| **Crisis** | Overdose, severe consequences, desperation | "Overdosed last night", "lost my job/family", "want to stop but can't" | Emergency — harm reduction + immediate clinical referral |
| **Recovery** | Active treatment, sobriety maintenance | "30 days clean", "in treatment", "meeting helped", "sponsor says..." | Support reinforcement — relapse prevention resources |

**Why this is novel:** While Prochaska's model (precontemplation → contemplation → preparation → action → maintenance) is well-established in clinical settings via structured interviews, SIGNAL operationalizes narrative stage detection on *unstructured social media text* using NLP. This is a methodological contribution — applying a clinical framework to a computational task.

**Why it's defensible under questioning:** The stages map directly to established addiction science. If a judge asks "where did these stages come from?", the answer is "Prochaska's Transtheoretical Model and NIDA's Cycle of Addiction, adapted for textual signal detection." That's a much stronger answer than "we made up four states for a Markov model."

---

## Research Landscape: Positioning SIGNAL's Novelty

**What exists (and what SIGNAL does differently):**

| Paper | What They Did | What SIGNAL Does Differently |
|---|---|---|
| **Lu et al. (2019)** — "Redditors in Recovery" (KDD) | Binary classifier predicting user transitions from casual drug forums → recovery forums. Cox regression for transition likelihood. | SIGNAL classifies **6 granular narrative stages per post**, not a binary forum-level transition. Stage classification is per-text, not per-user migration. |
| **Tamersoy et al. (2015)** — "Characterizing Smoking and Drinking Abstinence from Social Media" (ACM HT) | Used Reddit badge data to characterize long-term abstinence via linguistic features and network structure. | SIGNAL doesn't require self-reported badges. It classifies narrative stage from raw post text alone, enabling analysis on any dataset without user-level metadata. |
| **Valdez & Patterson (2022)** — Reddit addiction communities (PLOS Digital Health) | K-means clustering found 3 behavioral clusters: personal struggle, giving advice, seeking advice. | SIGNAL uses **theory-driven stages** from addiction science, not unsupervised clusters. The 6 stages map to clinical intervention points — not just behavioral patterns. |
| **Giyahchi et al. (2023)** — Post intent detection in online health support groups (Springer) | Fine-tuned language models to detect post **intents** (sharing, venting, requesting info) in smoking cessation groups. | SIGNAL classifies **where in the addiction arc** a person is, not what their post intends to do. A "sharing" post could be Experimentation or Crisis — the intent is the same, the clinical meaning is completely different. |
| **MacLean et al. (2015)** — "Forum77" (ACM CSCW) | Analyzed an online health forum dedicated to addiction recovery using qualitative and computational methods. | SIGNAL provides a **quantitative, multi-method classification pipeline** with pharmacological grounding. Forum77 was primarily qualitative analysis. |

**The gap SIGNAL fills:** No published work operationalizes Prochaska's Transtheoretical Model (or any stage-based addiction model) as a **multi-class text classification task** on social media posts with **three-method comparison** and **clinical grounding**. The closest work (Lu et al.) does binary transition prediction at the user level — not 6-stage classification at the post level.

**How to frame novelty in the report and under questioning:**
- DO say: "We adapt established clinical stage models for computational text classification — a task that, to our knowledge, has not been performed on social media substance abuse data."
- DO NOT say: "Nobody has ever thought of this." (Tamersoy and Lu were in the neighborhood.)
- DO cite Lu et al. and Tamersoy et al. as related work — showing awareness of the landscape signals sophistication, not ignorance.
- DO position SIGNAL as "extending binary transition detection to granular, theory-grounded stage classification with pharmacological context" — that's the precise novelty claim.

**Key references to include in the competition report:**
1. Prochaska & DiClemente (1983) — Transtheoretical Model
2. Lu et al. (2019) — Redditors in Recovery (KDD)
3. Tamersoy et al. (2015) — Characterizing Abstinence from Social Media
4. Valdez & Patterson (2022) — Reddit addiction help-seeking behaviors
5. NIDA Cycle of Addiction framework

---

## Architecture

```
INPUT
  Social media post (free text) OR batch corpus
       │
       ▼
┌─────────────────────────────────────────────────────┐
│          LAYER 1: SUBSTANCE RESOLUTION              │
│                                                     │
│   Slang Lexicon ──→ Clinical Entity Resolution      │
│   "blues" → "counterfeit oxycodone (fentanyl)"      │
│   "lean" → "codeine/promethazine"                   │
│   "bars" → "alprazolam (Xanax)"                     │
│                                                     │
│   3-method comparison:                              │
│   ├── Rule-based: curated slang→entity lexicon      │
│   ├── Embedding: SBERT + nearest-neighbor to known  │
│   │   substance descriptions                        │
│   └── LLM: Gemini zero-shot entity extraction       │
│   └── Ensemble: weighted agreement                  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│       LAYER 2: NARRATIVE STAGE CLASSIFICATION       │
│                                                     │
│   Post text → stage assignment with confidence      │
│                                                     │
│   3-method comparison:                              │
│   ├── Rule-based: stage-specific keyword patterns   │
│   │   + linguistic markers (tense, hedging, urgency)│
│   ├── Embedding: cosine similarity to stage         │
│   │   exemplar centroids (50 exemplars/stage)       │
│   └── LLM: Gemini few-shot with stage definitions   │
│   └── Ensemble: weighted agreement                  │
│                                                     │
│   Output: {stage, confidence, method_agreement}     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│     LAYER 3: CLINICAL RISK CONTEXTUALIZATION        │
│                                                     │
│   For each resolved substance:                      │
│   ├── FAISS/BM25 → retrieve relevant knowledge      │
│   │   chunks (pharmacology, warnings, interactions) │
│   ├── FAERS lookup → PRR, ROR, EBGM scores for     │
│   │   adverse events associated with substance      │
│   ├── Combination risk → flag dangerous poly-drug   │
│   │   patterns from knowledge base                  │
│   └── Stage-specific risk → clinical implications   │
│       of THIS substance at THIS narrative stage     │
│                                                     │
│   Output: ClinicalContext per substance             │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│     LAYER 4: EVIDENCE-GROUNDED ANALYST BRIEF        │
│                                                     │
│   Gemini synthesizes all layers into structured     │
│   brief with citations:                             │
│   • Substances detected (with clinical names)       │
│   • Narrative stage + supporting evidence            │
│   • Pharmacological risk factors (from KB + FAERS)  │
│   • Stage-appropriate intervention recommendation   │
│   • Confidence + method agreement summary           │
│                                                     │
│   Every claim cites a source:                       │
│   [KB-chunk-id] or [FAERS-signal-id] or [method]   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              DASHBOARD (Streamlit)                   │
│                                                     │
│   Page 1: Deep Analysis                             │
│     Paste text → full 4-layer report, live          │
│                                                     │
│   Page 2: Narrative Pulse                           │
│     Corpus-level stage distributions over time      │
│     Stacked area chart showing community shifts     │
│                                                     │
│   Page 3: Method Comparison                         │
│     Side-by-side eval for BOTH classification tasks │
│     Precision/Recall/F1 per method per task         │
└─────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
SIGNAL/
  signal/
    __init__.py
    config.py                         # Paths, Vertex AI config, constants, stage definitions
    
    ingestion/
      __init__.py
      post_ingester.py                # Post dataclass, dataset loaders, cleaning, batch embedding
      
    substance/
      __init__.py
      slang_lexicon.py                # Curated slang → clinical entity mapping (~200 entries)
      rule_based_detector.py          # Lexicon lookup + regex + context window
      embedding_detector.py           # SBERT similarity to substance description embeddings
      llm_detector.py                 # Gemini zero-shot substance extraction
      ensemble.py                     # Weighted voting + comparison metrics
      
    narrative/
      __init__.py
      stage_definitions.py            # 6 stages with exemplars, keywords, linguistic markers
      rule_based_classifier.py        # Keyword + tense + hedging + urgency pattern matching
      fine_tuned_classifier.py        # DistilBERT fine-tuned on stage-annotated exemplars
      llm_classifier.py               # Gemini few-shot stage classification
      ensemble.py                     # Weighted voting + comparison metrics
      train_distilbert.py             # Training script: data prep, fine-tuning, cross-validation
      
    grounding/
      __init__.py
      clinical_contextualizer.py      # FAISS/BM25 retrieval + FAERS lookup + combination risk
      
    synthesis/
      __init__.py
      brief_generator.py              # Gemini analyst brief with evidence citations
      pipeline.py                     # SIGNALPipeline: text → full 4-layer analysis
      
    temporal/
      __init__.py
      narrative_tracker.py            # Aggregate stage distributions over time windows
      spike_detector.py               # Rolling-average threshold for stage proportion shifts
      
    dashboard/
      __init__.py
      signal_app.py                   # Streamlit main entry
      pages/
        __init__.py
        deep_analysis.py              # Page 1: paste text → full report
        narrative_pulse.py            # Page 2: temporal stage distributions
        method_comparison.py          # Page 3: dual evaluation charts
        
    eval/
      __init__.py
      evaluator.py                    # Run all methods on labeled data, produce comparison tables
      
    tests/
      __init__.py
      test_setup.py                   # Validates imports, data paths, API connectivity
      test_substance_detection.py     # Known slang resolved, NegEx works, methods agree
      test_narrative_classification.py # Stage assignment correctness on exemplars
      test_grounding.py               # FAISS retrieval + FAERS lookup return data
      test_pipeline_e2e.py            # Full pipeline text → brief
      
  datasets/                           # Downloaded (gitignored)
    reddit_mh_rmhd/                    # Dataset #1: RMHD
    reddit_mh_labeled/                 # Dataset #2: Labeled
    reddit_mh_cleaned/                 # Dataset #3 or #4: best cleaned version
    uci_drug_reviews/                  # Dataset #5: UCI Drug Review (competition-recommended)
    depression_emo/                    # Dataset #8: DepressionEmo
    
  opioid_data/                        # Copied from opioid_track (confirmed working)
    knowledge_chunks/                  # 55 files
    faers_signal_results.json          # 204 signals
    opioid_pharmacology.json           # Receptor binding, LD50, potency
    opioid_mortality.json              # CDC data (for optional temporal validation)
```

---

## What We Reuse from opioid_track

| Asset | Status | How SIGNAL uses it |
|---|---|---|
| `data/knowledge_chunks/` (55 files) | ✅ Confirmed on disk | RAG corpus for Layer 3 clinical grounding |
| `data/faers_signal_results.json` | ✅ Confirmed loadable | Adverse event signal lookup in Layer 3 |
| `data/opioid_pharmacology.json` | ✅ Should be loadable | Receptor binding, LD50, potency for clinical context |
| FAISS + BM25 indexing pipeline | ✅ Confirmed working | Core retrieval for clinical grounding |
| `data/opioid_mortality.json` | ✅ Should be loadable | Optional: CDC trend backdrop on Narrative Pulse page |
| `dashboard/components/charts.py` | ⚠️ May need adaptation | Dark theme Plotly constants if reusable |
| `config.py` constants | ⚠️ Partial | Import term lists: `OPIOID_SAFETY_TERMS`, `MUST_INCLUDE_OPIOIDS` |

**NOT reused (and not needed):** OpioidWatchdog (doesn't run), CMS fetchers, NDC classifiers, MME mappers, geographic joiner, demographics builder. All of these are either broken or irrelevant. SIGNAL's clinical grounding is built fresh on top of the working data + retrieval assets.

---

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| LLM | Gemini 2.0 Flash (Vertex AI) | Confirmed working, GCP credits, fast, structured output |
| Embeddings (primary) | Vertex AI text-embedding-004 (768d) | Production quality, credits available |
| Embeddings (fallback) | sentence-transformers all-MiniLM-L6-v2 | Local CPU, zero cost if quota issues |
| **Stage classifier** | **DistilBERT (fine-tuned)** | **Real model training — HuggingFace Trainer on curated stage data** |
| Vector store | FAISS (faiss-cpu) | Already working in opioid_track |
| Sparse retrieval | BM25 (rank_bm25) | Already working in opioid_track |
| ML classification | scikit-learn (LogisticRegression, cosine_similarity) | Substance detection baseline |
| Dashboard | Streamlit | Existing infrastructure |
| Charts | Plotly | Existing dark theme potential |
| Data processing | pandas, numpy | Standard |

**New dependencies:** `sentence-transformers`, `google-cloud-aiplatform`, `google-generativeai`, `transformers`, `datasets`, `accelerate`

**NOT needed (removed vs SIREN):** `bertopic`, `ruptures`, `hdbscan`, `umap-learn` — none of which you've used before.

---

## Datasets (Official — from Dr. Lee's recommended list)

### Social Media Corpus (Primary)

| # | Dataset | Source | Key Features | SIGNAL Use |
|---|---|---|---|---|
| 1 | Reddit Mental Health Dataset (RMHD) | [Kaggle - entenam](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset) | Posts from multiple MH subreddits, large-scale | Primary corpus for ingestion, embedding, stage exemplar sourcing |
| 2 | Reddit Mental Health Data (Labeled) | [Kaggle - neelghoshal](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data) | Labeled (stress, depression) | Supervised classification baseline, distress signal features |
| 3 | Mental Health Reddit Dataset (Cleaned) | [Kaggle - cc2524](https://www.kaggle.com/datasets/cc2524/mental-health-reddit) | Preprocessed | Quick experiments, RAG pipeline testing |
| 4 | Reddit MH Cleaned Research Dataset | [Kaggle - dhanesha1210](https://www.kaggle.com/datasets/dhanesha1210/reddit-mental-health-cleaned-research-dataset) | Curated from multiple subreddits | LLM and embedding pipeline input |

**Strategy:** Download all 4 on Day 1. Inspect columns, size, timestamps, overlap. Pick best 2 as primary corpus. Priority: whichever has (a) subreddit labels AND (b) timestamps.

### Substance & Drug Text

| # | Dataset | Source | Key Features | SIGNAL Use |
|---|---|---|---|---|
| 5 | UCI Drug Review Dataset | [Kaggle - jessicali9530](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) | 200K+ reviews, drug names, conditions, ratings, **dates** | Substance detection ground truth, temporal analysis, embedding training |

**This is the competition's specifically recommended Kaggle dataset.**

### Public Health Ground Truth

| # | Dataset | Source | SIGNAL Use |
|---|---|---|---|
| 6 | NIDA Drug Use Trends | [nida.nih.gov](https://nida.nih.gov/research-topics/trends-statistics) | Population baselines for report context |
| 7 | CDC Drug Overdose Data | [data.cdc.gov](https://data.cdc.gov/) | Ground truth validation, Narrative Pulse backdrop (existing `opioid_mortality.json`) |

### Research-Level Optional

| # | Dataset | Source | SIGNAL Use |
|---|---|---|---|
| 8 | DepressionEmo | [GitHub](https://github.com/abuBakarSiddiqurRahman/DepressionEmo) | Multi-emotion labeled Reddit posts, fine-grained emotional signal features |

### Researcher-Created Data (Transparent, Disclosed)

| Data | Method | Count | Purpose |
|---|---|---|---|
| Narrative stage exemplars | Gemini-assisted curation of REAL posts from datasets #1-4, human-validated | 300 (50/stage) + ~300 augmented paraphrases | DistilBERT training + few-shot prompts |
| Expert-annotated validation set | Researcher manually labels real posts | 100 posts | Gold standard evaluation for stage classification |
| Slang resolution test cases | Researcher-written test inputs with known ground truth | 50 cases | Evaluation of slang→clinical resolution engine (only synthetic data) |

---

## Slang-to-Clinical Resolution: The Hidden Innovation

This is a real, unsolved problem in public health NLP. Most systems either:
- Miss street terminology entirely (CDC reports don't mention "percs" or "blues")
- Detect keywords without clinical resolution (flagging "lean" but not mapping it to codeine/promethazine + CYP2D6 metabolism risk)

SIGNAL's lexicon is a curated, clinically-grounded mapping. Sample entries:

```
OPIOIDS:
  "blues", "M30s", "pressies", "dirty 30s" → counterfeit oxycodone (likely fentanyl-contaminated)
  "oxy", "percs", "roxys"                  → oxycodone (OxyContin/Percocet)
  "lean", "purple drank", "sizzurp", "mud" → codeine/promethazine (CYP2D6 metabolized)
  "nod", "on the nod", "nodding out"       → opioid-induced sedation (active intoxication marker)
  "dope", "H", "boy", "brown"              → heroin (diacetylmorphine)
  "subs", "strips", "bupe"                 → buprenorphine (Suboxone — note: MAT indicator, NOT misuse)
  "tram", "ultram"                         → tramadol
  "china white"                            → fentanyl or fentanyl analogue
  
BENZODIAZEPINES:
  "bars", "xans", "zans", "sticks"         → alprazolam (Xanax)
  "kpins", "k-pins"                        → clonazepam (Klonopin)
  "vals"                                   → diazepam (Valium)
  
STIMULANTS:
  "addy", "addies"                         → amphetamine (Adderall)
  "ice", "tina", "glass"                   → methamphetamine
  "molly", "mandy"                         → MDMA
  
POLY-DRUG PATTERNS (HIGH RISK):
  "bars" + "lean"                          → benzodiazepine + opioid (respiratory depression risk)
  "speedball"                              → heroin + cocaine (cardiac + respiratory risk)
  "goofball"                               → heroin/fentanyl + methamphetamine
  
RECOVERY/MAT INDICATORS (classify differently):
  "on subs", "on the program"             → medication-assisted treatment (positive signal)
  "vivitrol", "naltrexone"                → opioid antagonist treatment
  "narcan", "naloxone"                    → harm reduction (may indicate proximity to crisis)
```

This lexicon (~200 entries) is a deliverable in itself. Judges can see it, evaluate it, and recognize it as domain expertise.

---

## Phased Implementation

### Phase 0: Foundation (Day 1, ~4 hours)

**Goal:** Everything installed, data accessible, project skeleton committed.

- [ ] Create `SIGNAL/` directory structure with all `__init__.py` files
- [ ] Copy confirmed-working opioid_track data: `knowledge_chunks/`, `faers_signal_results.json`, `opioid_pharmacology.json`, `opioid_mortality.json`
- [ ] Copy and adapt FAISS/BM25 indexing code from opioid_track (this already works — just reorganize imports)
- [ ] Install minimal new dependencies: `sentence-transformers`, `google-cloud-aiplatform`, `google-generativeai`
- [ ] Download all 8 datasets from Dr. Lee's recommended list:
  - Kaggle: RMHD (#1), Reddit MH Labeled (#2), MH Reddit Cleaned (#3), MH Cleaned Research (#4), UCI Drug Review (#5)
  - GitHub: DepressionEmo (#8)
  - Note: CDC data (#7) already available as `opioid_mortality.json`; NIDA (#6) is reference, not downloadable dataset
- [ ] **Day 1 dataset audit:** For each Reddit dataset, check: column names, row count, presence of timestamps, subreddit labels, any substance-specific labels. Document findings. Pick best 2 as primary corpus.
- [ ] Verify Vertex AI connection with test embedding + Gemini call
- [ ] Write `signal/config.py` with paths, Vertex AI project config, stage definitions
- [ ] Write `signal/tests/test_setup.py`

**Exit criterion:** `python -m signal.tests.test_setup` passes — imports work, Vertex AI responds, FAISS loads knowledge chunks, FAERS JSON loads, datasets on disk.

---

### Phase 1: Ingestion & Embedding (Days 2-3, ~8 hours)

**Goal:** Unified post corpus embedded and indexed.

#### 1a — Data Preprocessing (~3 hours)
- [ ] `Post` dataclass: `{id, text, timestamp, source, subreddit, label, raw_metadata}`
- [ ] Dataset-specific loaders:
  - Reddit MH datasets (#1-4): subreddit labels + post text (+ timestamps if available)
  - UCI Drug Review (#5): condition + rating + drug name + date (substance ground truth)
  - DepressionEmo (#8): emotion labels (supplementary)
- [ ] Text cleaning: URL removal, whitespace normalization, encoding fixes
- [ ] Merge into unified corpus with source tracking

#### 1b — Embedding & Indexing (~3 hours)
- [ ] Batch embed all posts with Vertex AI `text-embedding-004` (fallback: local SBERT)
- [ ] Build FAISS index with ID→metadata mapping
- [ ] Handle rate limits: batch in chunks of 250, with exponential backoff

#### 1c — Stage Exemplar Creation (Gemini-Assisted, ~5.5 hours)
- [ ] **Pass 1 — Gemini pre-filtering (~1.5h):** Take 1,000 random posts from corpus. Send to Gemini in batches of 20 with stage definitions. Filter to posts with confidence >0.8. This gives ~300-500 pre-sorted candidates.
- [ ] **Pass 2 — Human validation (~3h):** Review Gemini's high-confidence candidates. For each stage, scan ~80 candidates to accept 50 good exemplars. You're verifying, not searching — 10-15 seconds per review.
- [ ] **Pass 3 — Gap filling (~1h):** If any stage is under-represented (Crisis and Recovery typically have fewer Reddit posts), write 10-15 synthetic examples based on patterns from real posts. Disclose as "researcher-generated exemplars" in report.
- [ ] Embed exemplars and compute stage centroids
- [ ] Quick validation: nearest-neighbor search should return same-stage exemplars

**Exit criterion:** `index.search("I've been using again and can't stop", k=10)` returns relevant posts. Stage centroids computed. Exemplar set reviewed.

---

### Phase 2: Substance Resolution Engine (Days 4-6, ~12 hours)

**Goal:** Three methods for substance detection, fully compared. This is the competition's explicit requirement.

#### 2a — Slang Lexicon + Rule-Based Detector (~3 hours)
- [ ] Build `slang_lexicon.py`: ~200 slang→clinical entity mappings organized by drug class
- [ ] `rule_based_detector.py`:
  - Lexicon lookup with context window (±5 tokens)
  - NegEx-style negation detection ("I don't use...", "stopped taking...")
  - Regex patterns for dosage mentions, ROA (route of administration)
  - Poly-drug pattern flags (opioid + benzo → respiratory depression risk)
- [ ] Output: `{substances: [{slang_term, clinical_name, drug_class, negated: bool}], patterns: []}`

#### 2b — Embedding-Based Substance Detector (~4 hours)
- [ ] Create reference embeddings for each substance category using clinical descriptions
  - e.g., embed "oxycodone, a semi-synthetic opioid analgesic prescribed for moderate to severe pain"
- [ ] For input text: compute cosine similarity to all substance reference embeddings
- [ ] Train lightweight LogisticRegression on UCI Drug Review data (has drug name labels)
- [ ] Output: `{substances: [{name, confidence, method: "embedding"}]}`

#### 2c — LLM Substance Detector (~3 hours)
- [ ] Gemini 2.0 Flash with structured JSON output
- [ ] Few-shot prompt with slang resolution examples
- [ ] Output schema: `{substances: [{mentioned_term, clinical_name, drug_class, confidence}]}`
- [ ] Disk caching (hash of input text → cached response)
- [ ] Batch processing with rate limit management

#### 2d — Ensemble + Comparison + Evaluation (~4 hours)
- [ ] Weighted voting: rule=0.25, embedding=0.35, llm=0.40
- [ ] Agreement metric: % of substances detected by 2+ methods
- [ ] **Evaluation source 1 — UCI Drug Review ground truth:** Filter UCI dataset to opioid/benzo/stimulant posts. The `drugName` field is ground truth for substance presence. Measure precision/recall for each method against known drug labels.
- [ ] **Evaluation source 2 — Synthetic slang resolution test set:** Create 50 test cases with known slang→clinical mappings. Explicitly labeled as "synthetic evaluation set for slang resolution" in report. This tests the resolution engine, not the classifier — defensible like testing a spell-checker with known misspellings.
- [ ] Comparison table: precision/recall/F1 per method against both evaluation sources
- [ ] This becomes the first comparison dimension in the report

**Exit criterion:** "I've been popping blues and mixing with bars" → resolves to [counterfeit oxycodone/fentanyl, alprazolam], flagged as high-risk poly-drug pattern. All 3 methods produce output. Comparison table generated against both evaluation sources.

---

### Phase 3: Narrative Stage Classification (Days 7-10, ~18 hours)

**Goal:** Three methods for narrative stage assignment — including a fine-tuned transformer. This is SIGNAL's core innovation and the primary ML learning opportunity.

#### 3a — Stage Definitions & Rule-Based Classifier (~3 hours)
- [ ] `stage_definitions.py`: formal definitions with:
  - Keywords per stage (from addiction literature)
  - Linguistic markers: tense (past vs present vs future), hedging ("maybe", "thinking about"), urgency markers ("need", "can't", "have to"), recovery language ("clean", "sober", "meeting")
  - Negation handling: "I'm NOT using" ≠ Dependence
- [ ] `rule_based_classifier.py`:
  - Score each stage based on keyword presence + linguistic markers
  - Weighted scoring: keyword match (0.4) + linguistic pattern (0.3) + sentiment (0.3)
  - Output: `{stage, confidence, evidence_terms: []}`

#### 3b — Fine-Tuned DistilBERT Stage Classifier (~6 hours)

This is real model training — the centerpiece of SIGNAL's technical depth.

**Step 1 — Data augmentation (~1h):**
- [ ] Take 300 curated exemplars (50/stage) from Phase 1c
- [ ] Batch-paraphrase each exemplar once with Gemini: "Rewrite this post preserving the meaning and emotional tone but changing the wording." → ~600 training examples
- [ ] Split: 480 train / 120 validation (stratified by stage)
- [ ] Format as HuggingFace `Dataset`: `{"text": "...", "label": 0-5}`

**Step 2 — Fine-tuning script (~2.5h including learning the API):**
- [ ] Install: `pip install transformers datasets accelerate`
- [ ] Load `distilbert-base-uncased` with `DistilBertForSequenceClassification(num_labels=6)`
- [ ] Training arguments:
  ```
  learning_rate=2e-5
  num_train_epochs=10 (small dataset needs more epochs)
  per_device_train_batch_size=16
  weight_decay=0.01
  eval_strategy="epoch"
  save_strategy="epoch"
  load_best_model_at_end=True
  metric_for_best_model="f1_macro"
  ```
- [ ] Class-weighted loss: compute class weights from label distribution, pass to Trainer via custom `compute_loss` or use `class_weight` in a custom trainer subclass
- [ ] Metrics: macro F1, per-class precision/recall, accuracy
- [ ] Training time: ~5-10 minutes on CPU for 480 examples × 10 epochs

**Step 3 — Evaluation (~1.5h):**
- [ ] 5-fold cross-validation to get robust metrics (prevents overfitting claims)
- [ ] Evaluate on: held-out validation set + 100 expert-annotated posts (from Phase 3d)
- [ ] Generate confusion matrix: which stages get confused with each other?
- [ ] Compare vs random baseline and majority-class baseline
- [ ] Save best model checkpoint to disk for pipeline integration

**Step 4 — Pipeline integration (~1h):**
- [ ] `fine_tuned_classifier.py`: load saved model, tokenize input, predict stage + confidence
- [ ] Warm-start model on app launch (load once, predict many times — fast inference)
- [ ] Output: `{stage, confidence, logits_per_stage: []}`

**What you'll learn:** HuggingFace Trainer API, tokenization pipeline, class-weighted loss for imbalanced data, learning rate scheduling, overfitting diagnosis on small datasets, cross-validation with transformers, model checkpointing and inference.

**Why this matters for Technical Quality (40%):** The 3-method comparison becomes: heuristic patterns vs **trained transformer** vs LLM zero-shot. That's a genuine methods comparison where each approach has fundamentally different strengths and failure modes.

#### 3c — LLM Stage Classifier (~3 hours)
- [ ] Gemini few-shot with 2 exemplars per stage (12 total in prompt)
- [ ] Structured output: `{stage, confidence, reasoning, evidence_quotes: []}`
- [ ] Disk caching
- [ ] Batch processing

#### 3d — Ensemble + Comparison + Multi-Layer Evaluation (~6 hours)
- [ ] Weighted voting: rule=0.15, fine-tuned=0.45, llm=0.40
- [ ] Method agreement tracking
- [ ] **Evaluation Layer A — Inter-method agreement (primary metric):** Run all 3 methods on 500 posts. Compute Cohen's kappa / Fleiss' kappa measuring inter-method agreement. High agreement = converging evidence the task is well-defined. Disagreement patterns are findings: "Rule-based confuses Regular Use with Dependence due to keyword overlap, while LLM resolves via contextual reasoning."
- [ ] **Evaluation Layer B — Expert-annotated validation set (~4h of annotation):** Personally annotate 100 posts across all 6 stages. Disclose in report: "100 posts were expert-annotated by the researcher." This gives legitimate precision/recall/F1 numbers.
- [ ] **Evaluation Layer C — Face validity via demo:** Pre-select 3-4 demo examples covering unambiguous stages where any competent human would agree. Live demo IS validation.
- [ ] **Framing for report:** "In the absence of existing labeled datasets for addiction narrative stage detection — itself evidence of this task's novelty — we employ a multi-method convergence evaluation supplemented by expert annotation of 100 posts."
- [ ] Comparison table: per-stage precision/recall/F1 per method (against expert annotations)
- [ ] Confusion matrix per method (which stages get confused?)
- [ ] This is the second comparison dimension — giving judges TWO evaluation surfaces

**Exit criterion:** "I can't get through the day without it anymore, I'm sick every morning until I dose" → classified as Dependence with high confidence across all 3 methods. Comparison table shows meaningful inter-method differences. 100 expert annotations complete.

---

### Phase 4: Clinical Grounding + Synthesis (Days 10-12, ~10 hours)

**Goal:** The clinical intelligence layer that no other team can match.

#### 4a — Multi-Substance Knowledge Base + Clinical Contextualizer (~7 hours)

**4a-i — Expand knowledge base beyond opioids (~5h):**
The competition says "alcohol and drug abuse" — alcohol is mentioned first. Opioid-only grounding is a visible gap. Build lightweight knowledge bases for three additional substance classes:

- [ ] **Alcohol (~2h):** ~10 knowledge chunks covering: pharmacology (GABA-A receptor agonism, ADH/ALDH metabolism), toxicology (LD50, BAC thresholds, Wernicke-Korsakoff), withdrawal danger (delirium tremens, seizure risk — one of the few withdrawals that can be fatal), adverse event profile (curated from NIAAA data in same schema as FAERS), common medications (naltrexone, acamprosate, disulfiram). Use Gemini to draft from public NIAAA/WHO data, then review and correct.
- [ ] **Benzodiazepines (~1.5h):** ~8 knowledge chunks: pharmacology (GABA-A positive allosteric modulator), critical opioid+benzo interaction (respiratory depression — existing FAERS data IS relevant here since many signals involve concomitant benzo use), withdrawal risks (seizures, protracted withdrawal syndrome), common prescribing patterns.
- [ ] **Stimulants (~1.5h):** ~8 knowledge chunks: methamphetamine (dopamine/norepinephrine release, neurotoxicity), cocaine (dopamine reuptake inhibition, cardiac risk), prescription stimulants (Adderall diversion patterns). Key adverse events from literature.
- [ ] Create `supplementary_signals.json` with manually curated adverse event profiles for non-opioid substances, matching `faers_signal_results.json` schema
- [ ] Index all new chunks alongside existing 55 in FAISS/BM25 (indexer already works)

**Updated knowledge base inventory: ~81 chunks total (55 opioid + 10 alcohol + 8 benzo + 8 stimulant)**

**4a-ii — Clinical Contextualizer (~2h):**
- [ ] `clinical_contextualizer.py` — built fresh (NOT extending broken OpioidWatchdog):
  - Input: list of resolved substances from Layer 1
  - FAISS/BM25 hybrid query against ~81 knowledge chunks for each substance
  - FAERS signal lookup: load `faers_signal_results.json` + `supplementary_signals.json`, filter by substance
  - Pharmacology profile from `opioid_pharmacology.json` (opioids) or knowledge chunks (other classes)
  - Poly-drug interaction assessment: if multiple substances detected, check for known dangerous combinations
- [ ] Output: `ClinicalContext` object per substance

#### 4b — Analyst Brief Generator (~3 hours)
- [ ] `brief_generator.py`:
  - Gemini prompt assembles all layers into structured brief
  - Prompt includes: resolved substances, narrative stage, clinical context, method agreement
  - Output format:
    ```
    SIGNAL INTELLIGENCE BRIEF
    ─────────────────────────
    Substances Detected: [clinical names with resolution chain]
    Narrative Stage: [stage] (confidence: X%, method agreement: Y/3)
    
    Clinical Risk Assessment:
    [Substance-specific risks grounded in knowledge chunks + FAERS]
    
    Stage-Specific Implications:
    [What this substance + this stage means clinically]
    
    Recommended Response:
    [Stage-appropriate intervention — from prevention to emergency]
    
    Evidence Sources:
    [KB-chunk-id], [FAERS-signal-id], [method agreement data]
    ```
  - Citation enforcement: every factual claim must reference a knowledge chunk ID or FAERS signal

#### 4c — Pipeline Orchestration (~2 hours)
- [ ] `pipeline.py` — `SIGNALPipeline` class:
  - `analyze(text) → SignalReport` (single post, <5 seconds)
  - `analyze_batch(posts) → [SignalReport]` (corpus-level, with progress bar)
  - `get_method_comparison() → ComparisonReport` (eval metrics)
- [ ] Error handling: graceful fallback if Gemini rate-limited (return partial analysis without brief)

#### 4d — Evaluation Run (~1 hour)
- [ ] Run full pipeline on labeled test set (20% held-out)
- [ ] Generate comparison tables for both classification tasks
- [ ] Spot-check 10 analyst briefs for quality and citation accuracy

**Exit criterion:** `pipeline.analyze("I've been mixing lean with xans and I can't stop")` returns full 4-layer report with clinical context in <5 seconds. Analyst brief cites specific knowledge chunks and FAERS signals.

---

### Phase 5: Dashboard (Days 13-14, ~9 hours)

**Goal:** 3 polished pages that tell the SIGNAL story. Quality over quantity.

#### 5a — Core Layout + Deep Analysis Page (~4 hours)
- [ ] Streamlit app with dark theme (consistent with TruPharma brand if possible)
- [ ] **Deep Analysis page:**
  - Text input area (paste any social media post)
  - "Analyze" button → loading spinner → full 4-layer report display
  - Layer 1 display: resolved substances with slang→clinical mapping shown visually
  - Layer 2 display: narrative stage classification with confidence bars per method
  - Layer 3 display: clinical context cards — expandable sections for pharmacology, FAERS signals, interaction warnings
  - Layer 4 display: formatted analyst brief with highlighted citations
  - Method agreement indicator: "3/3 methods agree" vs "2/3 — see comparison"
- [ ] Pre-loaded example posts for demo (3-4 covering different stages and substances)

#### 5b — Narrative Pulse Page (~3 hours)
- [ ] **Narrative Pulse page:**
  - **CRITICAL: Check dataset timestamps in Phase 0.** Decision tree:
    - *If timestamps exist* (at least one major dataset has usable dates): Aggregate stage distributions by month, plot stacked area chart over time.
    - *If timestamps are absent/coarse:* **Pivot to cross-subreddit comparison.** Compare stage distributions ACROSS SUBREDDITS: "r/opiates is 35% Dependence, r/recovery is 60% Recovery, r/drugs is 45% Experimentation." This is arguably MORE defensible than temporal analysis since subreddit labels are ground truth.
  - Stacked area chart OR grouped bar chart (Plotly): x=time_window or subreddit, y=% of posts, color=narrative stage
  - Visual: when "Dependence" and "Crisis" proportions grow (or dominate), that's community escalation — visible at a glance
  - Optional overlay: CDC mortality trend line (from `opioid_mortality.json`) as ground truth backdrop
  - Substance filter: show stage distribution for posts mentioning specific substances
  - **Cross-subreddit framing for report:** "SIGNAL demonstrates that narrative stage distributions vary systematically across online communities, enabling targeted public health intervention — prevention messaging in experimentation-dominant communities, crisis resources in crisis-dominant communities."

#### 5c — Method Comparison Page (~2 hours)
- [ ] **Method Comparison page:**
  - Dual comparison: substance detection AND narrative stage classification
  - Per-method metrics: precision, recall, F1, accuracy — displayed as grouped bar charts
  - Confusion matrices for narrative stage classification (heatmaps)
  - Agreement analysis: when do methods disagree, and what does disagreement reveal?
  - This page directly addresses the competition's "compare multiple approaches" requirement with DOUBLE the evaluation surface area

**Exit criterion:** All 3 pages render cleanly. Demo flow: open app → paste text → see full analysis → flip to Narrative Pulse → flip to Method Comparison. Under 2 minutes, compelling at every step.

---

### Phase 6: Polish, Report & Submission (Days 15-16, ~8 hours)

#### 6a — End-to-End Testing (~2 hours)
- [ ] Run full pipeline on 100 diverse posts, verify no crashes
- [ ] Test edge cases: empty text, very long text, no substances detected, ambiguous stage
- [ ] Verify dashboard responsiveness and error handling
- [ ] Fix any pipeline failures

#### 6b — Competition Report (~3 hours)
- [ ] 4-page report structure:
  1. **Problem & Approach** (1 page): The narrative stage concept, why it matters, how it differs from binary classification. Related work positioning: cite Lu et al. (2019), Tamersoy et al. (2015), Valdez & Patterson (2022). Frame SIGNAL as extending binary transition detection to granular, theory-grounded stage classification.
  2. **Architecture & Methods** (1 page): 4-layer pipeline, 3-method comparison on 2 dimensions (including fine-tuned DistilBERT), clinical grounding via multi-substance KB
  3. **Results & Evaluation** (1 page): Comparison tables, confusion matrices, cross-validation metrics, inter-method agreement (kappa), example analyst briefs, narrative stage distributions across subreddits/time
  4. **Ethics & Impact** (1 page): Population-level only, no individual identification, privacy-preserving aggregation, clinical grounding prevents hallucinated medical claims, stage-appropriate intervention recommendations

#### 6c — Demo Preparation (~2 hours)
- [ ] Script the 2-minute demo video:
  - 0:00-0:20 — Problem framing (30 seconds of "why this matters")
  - 0:20-1:00 — Live Deep Analysis demo (paste a post, walk through the 4-layer output)
  - 1:00-1:30 — Narrative Pulse (show community-level stage shift)
  - 1:30-1:50 — Method Comparison (quick flash of dual evaluation)
  - 1:50-2:00 — Close (what this enables for public health)
- [ ] Record and edit demo video
- [ ] Final submission by noon April 6

#### 6d — Buffer (~1 hour)
- [ ] Reserved for fires

**Exit criterion:** Submission package complete — working deployed app, 4-page report, 2-minute demo video.

---

## Timeline Summary

| Phase | Days | Hours | Deliverable |
|---|---|---|---|
| 0 — Foundation | Day 1 | 5h | Working environment, data accessible, timestamp check |
| 1 — Ingestion + Exemplars | Days 2-3 | 11.5h | Embedded corpus, Gemini-assisted stage exemplars |
| 2 — Substance Resolution | Days 4-6 | 14h | 3-method substance detection + dual evaluation |
| 3 — Narrative Stages | Days 7-10 | 18h | **Fine-tuned DistilBERT** + rule-based + LLM + 100 expert annotations + multi-layer eval |
| 4 — Clinical Grounding | Days 11-13 | 15h | Multi-substance KB (81 chunks) + full 4-layer pipeline |
| 5 — Dashboard | Days 14-15 | 5h | 3-page polished demo (lean execution) |
| 6 — Polish | Day 16 | 3h | Report + video + submission |
| **Total** | **16 days** | **~71.5h** | **Deadline: April 6** |

At 4-5 hours/day over 16 days = 64-80 hours available. 71.5h is tight but inside the envelope. Phase 5 and 6 are lean — the pipeline must work by end of Phase 4.

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| Vertex AI quota exhaustion | Fallback to local SBERT for embeddings; aggressive disk caching for all Gemini calls |
| Stage exemplars are low quality | UCI Drug Review dataset has real patient narratives — mine these first. Hand-write only to fill gaps |
| Narrative stage classification is too hard | Even imperfect classification is novel. Show the confusion matrix — "Dependence vs Regular Use" confusion is itself an interesting finding |
| **DistilBERT overfits on 480 examples** | 5-fold cross-validation, early stopping, class-weighted loss, Gemini paraphrase augmentation (2x data). If overfitting persists, report honest metrics — imperfect results + solid methodology > inflated numbers |
| **DistilBERT training fails on CPU** | 480 examples trains in <10 min on CPU. If truly stuck, fall back to LogisticRegression on SBERT embeddings (original plan) — still have rule-based + LLM comparison |
| FAISS/BM25 retrieval returns irrelevant chunks | Existing pipeline is confirmed working; test with known queries early in Phase 0 |
| Dashboard scope creep | HARD LIMIT: 3 pages. No 4th page. Spend saved time on polish |
| Gemini rate limits during demo | Pre-cache analysis results for 5 demo examples; live analysis is bonus, not requirement |
| Report takes too long | Template the structure in Phase 0; fill in results as they're generated |

---

## What Makes This Beat Agentic IDE Teams

A team of 3 using Cursor/Claude Code can scaffold a generic "classify posts with LLM → show dashboard" system in a weekend. Here's why SIGNAL still wins:

1. **They don't have the data.** 55+ knowledge chunks, 204 FAERS signals, pharmacological profiles — this took months of the opioid track project to assemble. No amount of agentic coding generates domain-specific clinical knowledge bases overnight.

2. **They won't think of narrative stages.** This requires addiction science domain knowledge. Code generators produce *code*, not *conceptual frameworks*. The narrative stage idea is the kind of insight that comes from sustained engagement with the problem domain.

3. **They won't train a model on a novel task.** Agentic IDEs can scaffold a HuggingFace fine-tuning script, but they can't create the curated, theory-grounded training data that makes the task meaningful. SIGNAL's DistilBERT is trained on researcher-curated, Prochaska-informed stage annotations. That's domain expertise encoded as training data.

4. **Their clinical grounding will be shallow.** They might call an API or scrape a webpage. SIGNAL's grounding passes through a curated, indexed knowledge base with pharmacological precision (receptor binding affinities, LD50 values, FAERS signal scores). That depth is visible in the output.

5. **They'll have one comparison dimension; SIGNAL has two.** The competition requires comparing methods. Most teams will compare methods on one task. SIGNAL compares on two (substance detection + narrative stage), with the narrative stage comparison featuring a **trained transformer** vs rule-based vs LLM — a genuine methods comparison.

6. **SIGNAL's demo is interactive and immediate.** Paste any text → get a rich multi-layered analysis. Judges can try their own examples. A scaffolded dashboard with pre-computed charts can't do this.

---

## Evaluation Criteria Mapping

| Criterion | Weight | How SIGNAL addresses it |
|---|---|---|
| **Technical quality** | 40% | 4-layer pipeline, **fine-tuned DistilBERT**, dual 3-method comparison (with trained model), FAISS/BM25 hybrid RAG, evidence-cited briefs, 5-fold cross-validation, confusion matrices, inter-annotator agreement analysis |
| **Innovation** | 30% | Narrative stage classification (novel task), slang-to-clinical resolution engine, dual-dimension method comparison, stage-specific clinical risk assessment, multi-substance knowledge base |
| **Impact & relevance** | 20% | Analyst-ready briefs actionable by public health workers, narrative stage tracking reveals community escalation, pharmacologically grounded (not hallucinated), covers alcohol + opioids + benzos + stimulants |
| **Clarity & communication** | 10% | 3-page focused dashboard, interactive live demo, 4-page report with clear structure, related work positioning |
