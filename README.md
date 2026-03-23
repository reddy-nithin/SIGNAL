# SIGNAL — Substance Intelligence through Grounded Narrative Analysis of Language

> **NSF NRT Project Challenge 1** · UMKC 2026 Spring Research-A-Thon
> AI for Substance Abuse Risk Detection from Social Signals

---

## What is SIGNAL?

SIGNAL is a multi-layered AI system that goes beyond binary risk classification of social media posts. Instead of asking *"Is this post risky?"*, SIGNAL asks:

**"What chapter of the addiction story is this person living — and what does clinical science say about what comes next?"**

For any social media post, SIGNAL produces a clinical intelligence report covering:

- **Substance Resolution** — Street slang → clinical entity (e.g., "blues" → counterfeit oxycodone, likely fentanyl-contaminated)
- **Narrative Stage Classification** — Where in the addiction arc: Curiosity → Experimentation → Regular Use → Dependence → Crisis → Recovery
- **Clinical Risk Contextualization** — Pharmacological grounding via 80+ knowledge chunks and 265 FAERS adverse event signals
- **Evidence-Cited Analyst Brief** — Actionable intelligence for public health workers, with every claim sourced

## Architecture

```
┌─────────────────────────────────────────┐
│   LAYER 1: SUBSTANCE RESOLUTION         │
│   Slang → Clinical Entity               │
│   3 methods: Rule-based │ Embedding │ LLM│
├─────────────────────────────────────────┤
│   LAYER 2: NARRATIVE STAGE              │
│   Post → Addiction Stage (6 classes)     │
│   3 methods: Rule-based │ DistilBERT│LLM│
├─────────────────────────────────────────┤
│   LAYER 3: CLINICAL GROUNDING           │
│   FAISS/BM25 RAG over knowledge base    │
│   + FAERS pharmacovigilance signals     │
├─────────────────────────────────────────┤
│   LAYER 4: ANALYST BRIEF                │
│   Evidence-cited clinical report         │
└─────────────────────────────────────────┘
```

## Narrative Stages

SIGNAL's core innovation — classifying social media posts into clinically-grounded addiction narrative stages based on Prochaska's Transtheoretical Model and NIDA's Cycle of Addiction:

| Stage | Description | Example Signal |
|---|---|---|
| **Curiosity** | Pre-use interest | *"What does oxy feel like?"* |
| **Experimentation** | First/early use | *"Tried it last weekend, not a big deal"* |
| **Regular Use** | Patterned use | *"I use every Friday to unwind"* |
| **Dependence** | Compulsive use, withdrawal | *"Can't function without it"* |
| **Crisis** | Overdose, severe consequences | *"Overdosed last night"* |
| **Recovery** | Treatment, sobriety | *"30 days clean today"* |

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini 2.0 Flash (Vertex AI) |
| Embeddings | Vertex AI text-embedding-004 |
| Stage Classifier | Fine-tuned DistilBERT |
| Vector Store | FAISS + BM25 hybrid retrieval |
| Dashboard | Streamlit |
| Charts | Plotly |

## Knowledge Base

SIGNAL is grounded in real pharmacological data — not LLM hallucinations:

- **58 opioid knowledge chunks** — receptor binding affinities, LD50 values, FDA safety labels, CYP interactions
- **265 FAERS adverse event signals** — with PRR, ROR, and EBGM scores from FDA pharmacovigilance data
- **Alcohol, benzodiazepine, and stimulant knowledge chunks** — extending coverage beyond opioids

## Key Differentiators

- **Narrative stage classification** is a novel NLP task — no published work operationalizes Prochaska's model as a multi-class text classifier on social media
- **Dual 3-method comparison** — substance detection AND narrative stage classification, each evaluated with rule-based vs trained transformer vs LLM
- **Slang-to-clinical resolution engine** — 200+ street term mappings with poly-drug pattern detection
- **Fine-tuned DistilBERT** — real model training, not just API calls
- **Evidence-cited outputs** — every claim in analyst briefs traces to a knowledge chunk or FAERS signal

## Datasets

Per NSF NRT Challenge 1 guidelines:

- Reddit Mental Health Dataset (RMHD) — primary social media corpus
- UCI Drug Review Dataset — substance detection ground truth
- CDC Drug Overdose Data — public health validation
- DepressionEmo — emotional signal features
- NIDA Drug Use Trends — population baselines

## Project Structure

```
SIGNAL/
  signal/
    config.py                    # Paths, Vertex AI config, stage definitions
    ingestion/                   # Post dataclass, dataset loaders, embedding
    substance/                   # Slang lexicon, 3-method substance detection
    narrative/                   # Stage definitions, DistilBERT, 3-method classification
    grounding/                   # FAISS/BM25 retrieval, FAERS lookup, clinical context
    synthesis/                   # Gemini analyst briefs, pipeline orchestration
    temporal/                    # Stage distribution tracking over time/subreddits
    dashboard/                   # Streamlit app (3 pages)
    eval/                        # Evaluation framework, comparison tables
    tests/                       # Unit and integration tests
  opioid_data/                   # Pharmacological knowledge base
  datasets/                      # Downloaded social media datasets (gitignored)
  models/                        # Fine-tuned DistilBERT checkpoints (gitignored)
  evidence/                      # Phase completion evidence and metrics
```

## Dashboard

Three focused pages:

1. **Deep Analysis** — Paste any text → full 4-layer clinical intelligence report
2. **Narrative Pulse** — Community-level stage distributions across subreddits/time
3. **Method Comparison** — Side-by-side evaluation of all 3 methods on both tasks

## Ethics

- Population-level insights only — no individual identification
- Privacy-preserving aggregation of anonymized public data
- Clinical grounding prevents hallucinated medical claims
- Stage-appropriate intervention recommendations (prevention for Curiosity, emergency resources for Crisis)

## References

1. Prochaska, J.O. & DiClemente, C.C. (1983). Transtheoretical Model of Behavior Change
2. Lu, D. et al. (2019). Redditors in Recovery: Text Mining Reddit to Investigate Transitions into Drug Addiction. *KDD*
3. Tamersoy, A. et al. (2015). Characterizing Smoking and Drinking Abstinence from Social Media. *ACM HT*
4. Valdez, D. & Patterson, M. (2022). Computational analyses identify addiction help-seeking behaviors on Reddit. *PLOS Digital Health*

## Author

Developed as a solo project for the NSF NRT Project Challenge at UMKC, 2026 Spring Research-A-Thon.

---

*SIGNAL treats social media as a clinical signal source — transforming noisy online text into pharmacologically grounded intelligence.*
