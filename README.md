# Fake News Detection — Document Claim-Support Assessment Tool

A full-stack application that analyzes PDF documents and raw text by extracting claims, scoring each one with a machine-learning classifier, and optionally layering on a gated RoBERTa transformer signal and a real-world web-search fact-check verdict. Built for Use Case #15 (GenAI & Security).

**Scores are model-based estimates, not definitive truth.** They indicate pattern similarity to known credible or non-credible news sources. Use as a triage signal for human review, not as an automated verdict.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Directory Structure](#directory-structure)
- [ML Pipeline](#ml-pipeline)
  - [Training (Offline)](#training-offline)
  - [Inference (Online)](#inference-online)
  - [Assessment Labels](#assessment-labels)
  - [Dateline Leakage Ablation](#dateline-leakage-ablation)
  - [Client-Side JS Model](#client-side-js-model)
- [LIAR Ensemble (Gated Transformer)](#liar-ensemble-gated-transformer)
- [Agentic Fact-Check System](#agentic-fact-check-system)
- [PDF Processing Pipeline](#pdf-processing-pipeline)
- [Backend API](#backend-api)
  - [Endpoints](#endpoints)
  - [Request/Response Schemas](#requestresponse-schemas)
- [Frontend](#frontend)
  - [Components](#components)
  - [UI Features](#ui-features)
  - [Styling & Theme](#styling--theme)
- [Legacy Dashboard](#legacy-dashboard)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Dataset](#dataset)
  - [Environment Variables](#environment-variables)
- [Running](#running)
- [Model Performance](#model-performance)
  - [Held-out Test Set](#held-out-test-set)
  - [Confusion Matrix (Linear SVM)](#confusion-matrix-linear-svm)
  - [Per-Category Accuracy](#per-category-accuracy)
  - [Corpus Trend Data](#corpus-trend-data)
- [Known Limitations](#known-limitations)
- [Responsible Use](#responsible-use)

---

## Project Overview

This project addresses the problem of automated fake news detection by building a production-ready classifier that scores individual claims extracted from PDF documents. The core approach:

1. **Train** a TF-IDF + Linear SVM classifier on the Kaggle Fake and Real News Dataset (~44,900 articles)
2. **Serve** predictions via a FastAPI REST API with confidence scores
3. **Layer** additional signals on top of the base classifier:
   - A **gated RoBERTa transformer** (LIAR dataset, 6-class truthfulness) for "unverifiable" detection
   - An **agentic web-search fact-checker** (Wikipedia + Groq LLM) for real-world truth verification
4. **Present** results through a modern React frontend with PDF upload, text analysis, and per-claim scoring

The system explicitly never overrides the base classifier's label — secondary signals are always additive, and every prediction includes a disclaimer about the model's limitations.

---

## Architecture Overview

```
                         ┌─────────────────────────────────────────┐
                         │           React Frontend (Vite)         │
                         │   http://localhost:5173                  │
                         └────────────┬────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
              POST /predict    POST /analyze-pdf   GET /stats/*
                    │                 │                  │
                    ▼                 ▼                  │
┌──────────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (api/main.py)                    │
│                  http://localhost:8000                            │
│                                                                  │
│  ┌──────────┐  ┌─────────────────┐  ┌────────────────────────┐  │
│  │ Vectorizer│  │  Linear SVM     │  │  LIAR Ensemble Gate    │  │
│  │ (30k feat)│  │  (best_model)   │  │  (RoBERTa, gated)     │  │
│  └──────────┘  └─────────────────┘  └────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              PDF Processing Pipeline                      │   │
│  │  PyPDF2 → claim splitting → per-claim scoring            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Agentic Fact-Checker                         │   │
│  │  Wikipedia API + Groq compound-mini web_search → LLM verdict│   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Request Flow for PDF Analysis

```
User uploads PDF
       |
       v
[React Frontend] --POST /analyze-pdf--> [FastAPI Backend]
       |                                       |
       v                                       v
 Results displayed             PyPDF2 extracts text
 per claim with                from all pages
 color-coded scores                    |
                                       v
                              Text split into
                              sentence-level claims
                                       |
                                       v
                              Each claim cleaned
                              with shared clean_text()
                                       |
                                       v
                              TF-IDF vectorizer
                              (30,000 features)
                                       |
                                       v
                              Linear SVM classifier
                              returns P(fake), P(real)
                                       |
                                       v
                              Assessment mapped:
                              supported / unsupported /
                              uncertain / needs_review
                                       |
                                       v
                              JSON response with
                              per-claim results
                              + summary statistics
```

---

## Tech Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Runtime | Python | 3.9+ | Core language |
| API Framework | FastAPI | >= 0.110 | REST API server |
| Server | Uvicorn | >= 0.29 | ASGI server |
| ML Library | scikit-learn | >= 1.3 | TF-IDF vectorization, classifiers |
| PDF Parser | PyPDF2 | >= 3.0 | Text extraction from PDFs |
| Serialization | joblib | >= 1.3 | Model persistence |
| Data Processing | pandas | >= 2.0 | Dataset loading and manipulation |
| Numerical | numpy | >= 1.24 | Array operations |
| Request Validation | Pydantic | >= 2.0 | API request/response models |
| HTTP Client | requests | >= 2.31 | Groq API and Wikipedia API calls |
| Deep Learning | torch | >= 2.2 | RoBERTa transformer inference |
| Transformers | transformers | >= 4.40 | Hugging Face model loading |
| Multipart | python-multipart | >= 0.0.6 | File upload handling |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| UI Framework | React | 18.3+ | Component-based UI |
| Build Tool | Vite | 6.0+ | Dev server and bundler |
| Styling | Tailwind CSS | 4.0+ | Utility-first CSS |
| HTTP Client | Axios | 1.7+ | API communication |
| Icons | Lucide React | 0.400+ | SVG icon library |
| Language | JSX/JavaScript | ES2020+ | Component logic |

---

## Directory Structure

```
Fake-News-Detection1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusions (myenv/, *.safetensors)
├── key                                # API key file (gitignored content)
│
├── api/                               # Backend API
│   ├── main.py                        # FastAPI app, all endpoints, CORS, Pydantic models
│   └── pdf_processor.py               # PDF text extraction, claim splitting, scoring
│
├── src/                               # Training pipeline & shared code
│   ├── preprocessing.py               # Shared clean_text(), STOPWORDS, dataset loading
│   ├── train.py                       # Trains 4 classifiers, selects best, saves artifacts
│   ├── liar_ensemble.py               # Gated RoBERTa ensemble (LIAR 6-class)
│   ├── fact_check.py                  # Agentic web-search fact-checker (Wikipedia + Groq)
│   ├── export_js_model.py             # Exports compact 3,500-term LR for browser scoring
│   ├── build_dashboard.py             # Generates dashboard/index.html from template
│   └── ablation_dateline.py           # Measures dateline-leakage artifact effect
│
├── frontend/                          # React frontend
│   ├── package.json                   # Node.js project config
│   ├── package-lock.json              # Dependency lockfile
│   ├── vite.config.js                 # Vite config: React + Tailwind + API proxy
│   ├── index.html                     # HTML entry point (Google Fonts)
│   ├── .gitignore                     # Excludes node_modules/, dist/
│   └── src/
│       ├── main.jsx                   # React entry point, mounts <App />
│       ├── App.jsx                    # Root component: tabs, state, layout
│       ├── api.js                     # Axios instance (base URL, timeout)
│       ├── index.css                  # Tailwind import + CSS custom properties
│       └── components/
│           ├── Header.jsx             # Title, subtitle, model badge
│           ├── TextInput.jsx          # Text analysis: mode selector, textarea, results
│           ├── PdfUploader.jsx        # Drag-and-drop PDF upload with progress
│           ├── ResultsTable.jsx       # Sortable per-claim results table
│           ├── SummaryStats.jsx       # Aggregate stats + stacked bar chart
│           ├── FactCheckPanel.jsx     # Agentic fact-check verdict display
│           └── EnsemblePanel.jsx      # LIAR ensemble signal display
│
├── models/                            # Serialized models & artifacts
│   ├── best_model.joblib              # Production model (Linear SVM + CalibratedClassifierCV)
│   ├── vectorizer.joblib              # Fitted TfidfVectorizer (30,000 features)
│   ├── best_model_name.txt            # Plain text: "Linear SVM"
│   └── liar_v3/                       # LIAR-trained transformer
│       ├── manifest.json              # Model card: eval_f1_macro=0.445
│       ├── metadata.json              # Status, max_length, dataset info
│       ├── label_mapping.json         # 6-class label ↔ ID mapping
│       ├── model/                     # RoBERTa weights (safetensors)
│       └── tokenizer/                 # BPE tokenizer files
│
├── reports/                           # Metrics & exported artifacts
│   ├── model_comparison.json          # Accuracy/F1/precision/recall per model
│   ├── confusion_matrices.json        # 2×2 confusion matrix per model
│   ├── category_breakdown.json        # Per-subject-category accuracy
│   ├── trend_data.json                # Monthly fake/real volume (2015–2018)
│   ├── dateline_leakage_ablation.json # Before/after dateline stripping
│   └── js_model.json                  # Compact 3,500-term LR for browser fallback
│
├── dashboard/                         # Legacy static dashboard
│   ├── template.html                  # Source template (edit this)
│   └── index.html                     # Generated self-contained dashboard
│
├── data/                              # Dataset (not bundled)
│   └── README.md                      # Download instructions for Kaggle CSVs
│
├── docs/                              # Documentation
│   ├── limitations.md                 # Limitations & responsible-use note
│   └── DIRECTORY_STRUCTURE.md         # File-by-file directory reference
│
└── myenv/                             # Python virtual environment (gitignored)
```

---

## ML Pipeline

### Training (Offline, One-Time)

The training pipeline is implemented in `src/train.py` and produces all model artifacts and reports.

**Step 1: Data Loading & Cleaning** (`src/preprocessing.py`)

- Loads `data/Fake.csv` and `data/True.csv` from the Kaggle "Fake and Real News Dataset"
- Labels: `Fake.csv` → `label=0`, `True.csv` → `label=1`
- Merges into a single DataFrame (~44,900 rows)
- Parses dates with mixed format handling
- Concatenates `title` + `. ` + `text` into a `content` column
- Applies `clean_text()` with dateline stripping enabled

**Step 2: Text Preprocessing** (`src/preprocessing.py:clean_text()`)

The `clean_text()` function is shared between training and inference to prevent train/serve skew:

```
1. Strip Reuters dateline (regex: CITY (Reuters) — )
2. Remove URLs (http://..., www....)
3. Lowercase all text
4. Remove standalone "Reuters" mentions
5. Remove non-alphabetic characters
6. Collapse whitespace
```

**Step 3: Feature Extraction**

- `TfidfVectorizer` with:
  - `max_features=30,000` (vocabulary size)
  - `ngram_range=(1, 2)` (unigrams + bigrams)
  - `min_df=5` (minimum document frequency)
  - `sublinear_tf=True` (apply sublinear TF scaling)
  - Custom stopword list (hand-maintained, ~300 English stopwords)
- Produces sparse TF-IDF matrices: X_train shape = (35,911, 30,000)

**Step 4: Model Training & Selection**

Four classifiers trained on an 80/20 stratified split (seed=42):

| Classifier | Class | Key Parameters |
|-----------|-------|----------------|
| Logistic Regression | `LogisticRegression` | `max_iter=2000, n_jobs=-1` |
| Multinomial Naive Bayes | `MultinomialNB` | defaults |
| Linear SVM | `LinearSVC` → `CalibratedClassifierCV(cv=3)` | `max_iter=5000` |
| Random Forest | `RandomForestClassifier` | `n_estimators=200, n_jobs=-1` |

The Linear SVM is wrapped in `CalibratedClassifierCV` to produce `predict_proba()` outputs needed for confidence scores.

Best model selected by **weighted F1 score** on the held-out test set.

**Step 5: Artifact Persistence**

- `models/best_model.joblib` — serialized production model
- `models/vectorizer.joblib` — fitted TF-IDF vectorizer
- `models/best_model_name.txt` — plain text model name
- `reports/model_comparison.json` — all metrics per model
- `reports/confusion_matrices.json` — confusion matrices
- `reports/category_breakdown.json` — per-subject accuracy
- `reports/trend_data.json` — monthly article volume

### Inference (Online, At Request Time)

1. Same `clean_text()` function used in training (shared code, prevents skew)
2. Text vectorized with the same fitted `TfidfVectorizer`
3. `model.predict_proba()` returns `[P(fake), P(real)]`
4. For PDF analysis: each sentence/claim scored independently
5. For ensemble endpoint: LIAR transformer signal evaluated in parallel
6. For fact-check endpoint: Wikipedia + Groq web search + LLM verdict

### Assessment Labels

| Label | Criteria | Meaning |
|-------|----------|---------|
| `supported` | confidence >= 0.75 (or P(real) >= 0.85) | Text patterns closely match credible news sources |
| `unsupported` | confidence <= 0.25 (or P(fake) >= 0.85) | Text patterns align more with non-credible sources |
| `uncertain` | 0.25 < confidence < 0.75 | Model cannot strongly classify; manual review recommended |
| `needs_review` | cleaned text is empty | Too short, all stopwords, or non-English content |

### Dateline Leaking Ablation

Measured effect of Reuters dateline on model accuracy (`src/ablation_dateline.py`):

| Setting | Accuracy | Weighted F1 | Top Terms Favoring "Real" |
|---------|----------|-------------|---------------------------|
| Dateline left in (raw text) | 0.9926 | 0.9926 | `reuters`, `said`, `washington reuters`, `washington` |
| Dateline stripped (production) | 0.9881 | 0.9881 | `said`, `president donald`, `wednesday`, `thursday` |

**Accuracy delta: 0.45 percentage points.** Stripping the dateline removes the single biggest shortcut. The residual accuracy is attributable to genuine linguistic/stylistic signal, though some Reuters house style influence remains (formal attribution, weekday-dated reporting).

### Client-Side JS Model

`src/export_js_model.py` exports a compact **3,500-term Logistic Regression** model as JSON for in-browser fallback scoring. This is a *demo artifact* distinct from the production model:

- Vocabulary: 3,500 unigrams only (vs. 30,000 unigrams+bigrams in production)
- Model: Logistic Regression (vs. Linear SVM in production)
- Exported as: intercept + term → [idf, coefficient] mapping
- Size: compact enough to embed inline in HTML
- Used by: `dashboard/index.html` for offline predictions when backend is unreachable

---

## LIAR Ensemble (Gated Transformer)

The LIAR ensemble (`src/liar_ensemble.py`) adds an "unverifiable / disputed" signal on top of the binary TF-IDF classifier using a fine-tuned RoBERTa-base model.

### Why This Exists

The production TF-IDF model is trained on binary fake/real news articles. It has no way to represent a statement that is neither clearly true nor clearly false — e.g. "Donald Trump may resign in October" is a prediction, not a factual claim. LIAR's 6-class scheme (pants-fire → true) has a natural middle ground (barely-true / half-true) that maps onto "unverifiable, don't force a binary call."

### Quality Gate

The bundled classifier-v3.0 model reports `eval_f1_macro=0.445` on 6 classes (chance = 1/6 ≈ 0.167). A module-level quality gate (`MIN_F1_MACRO_TO_TRUST = 0.35`) evaluates the model's own reported validation F1:

- If the model clears the gate → `liar_model.trusted = True`, transformer loaded lazily on first prediction
- If the model fails the gate → `liar_signal()` always returns `None`, ensemble endpoint falls back to TF-IDF-only

The bundled v3.0 model clears this gate (`eval_f1_macro=0.445 > 0.35`).

### Label Buckets

LIAR's 6 truthfulness grades are collapsed into 3 buckets for the ensemble:

| LIAR Label | Bucket | Meaning |
|-----------|--------|---------|
| `pants-fire`, `false` | `leans-fake` | Aligns with fake class |
| `barely-true`, `half-true` | `uncertain` | Middle ground — triggers `unverifiable` flag |
| `mostly-true`, `true` | `leans-real` | Aligns with real class |

### Lazy Loading

The transformer (`torch`, `transformers`) is only imported if:
1. The quality gate passes AND
2. A prediction is actually requested

This means a deployment with no `models/liar_v3/` directory or a low-scoring model never pays the heavyweight dependency cost.

---

## Agentic Fact-Check System

The fact-check system (`src/fact_check.py`) provides real-world, evidence-grounded verdicts on top of the style classifier.

### Why This Exists

The TF-IDF model answers "does this text's style resemble credible/non-credible sources" — NOT "is this claim true right now." A short, factually-true statement like "donald trump is the president of USA" can score confidently "fake" on style grounds alone. The fact-check module gives the system that missing capability.

### Design: Search-Then-Generate (Not Agentic)

```
1. SEARCH STEP (cheap, tightly-bounded LLM tokens):
   ├── Wikipedia OpenSearch (free, keyless) → 1 authoritative reference
   └── groq/compound-mini, restricted to its web_search tool only
       (via compound_custom.tools.enabled_tools) → up to 2 recent articles

2. GENERATE STEP (one plain LLM call, no tools):
   └── Hand claim + snippets to openai/gpt-oss-120b
       → Structured JSON verdict
```

Groq's standalone `web_search` tool (the one returning clean `{title, url, content}` results)
only exists on the compound systems (`groq/compound`, `groq/compound-mini`) — it can't be
attached to a plain chat model via the `tools` param. Plain models like the `openai/gpt-oss-*`
family instead expose a *different* tool, `browser_search`, which drives an interactive
browsing session and returns an unstructured cited answer rather than a snippet list. Compound
systems do run their own server-side agentic loop, so token cost isn't as tightly bounded as a
plain non-agentic call; restricting `enabled_tools` to `["web_search"]` only (no
`code_interpreter`, no `visit_website`) keeps that loop as narrow as possible, and `MAX_TOKENS`
still caps the generation step's own request size, which is fully deterministic.

### Verdict Types

| Verdict | Meaning |
|---------|---------|
| `TRUE` | Claim is supported by reference snippets |
| `FALSE` | Claim contradicts reference snippets |
| `MISLEADING` | Technically true but deceptively framed, or missing important context |
| `UNVERIFIED` | Snippets don't give enough information — a legitimate answer, not a failure |

### Safety & Failsafe

- API key read from `GROQ_API_KEY` environment variable at call time
- Missing key → `available=False` with human-readable reason, never throws at import time
- Hard timeouts: `REQUEST_TIMEOUT_S = 20` seconds per HTTP call
- Bounded token cost: `MAX_CLAIM_CHARS=500`, `MAX_SNIPPET_CHARS=400`, `MAX_SNIPPETS=3`, `MAX_TOKENS=512`
- HTTP 401/413/429 errors handled gracefully with fallback messages
- JSON parsing failures → `available=False`, never a 500
- TF-IDF label/confidence always returned unchanged alongside fact-check result

---

## PDF Processing Pipeline

Implemented in `api/pdf_processor.py`, this pipeline processes uploaded PDF documents end-to-end.

### Step 1: Text Extraction

```python
PyPDF2.PdfReader(BytesIO(file_bytes))
```
- Iterates all pages, calls `page.extract_text()`
- Joins pages with double newline spacing
- Returns empty string if no text extractable (scanned/image-based PDF)

### Step 2: Claim Splitting

```python
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
```
- Primary: split on sentence boundaries (period/exclamation/question + whitespace)
- Fallback: if no sentence punctuation found (bullet lists, fragments), split on line breaks
- Filter: segments shorter than 10 characters are removed

### Step 3: Per-Claim Scoring

Each segment undergoes:
1. Cleaned with `clean_text()` (same function used in training)
2. Vectorized with production `TfidfVectorizer`
3. Classified with `model.predict_proba()` → P(fake), P(real)
4. Assessment mapped based on thresholds:
   - P(real) >= 0.85 → `supported`
   - P(fake) >= 0.85 → `unsupported`
   - Otherwise → `uncertain`
5. Empty cleaned text → `needs_review` (confidence = 0.0)

### Step 4: Summary Computation

Aggregate statistics computed across all scored segments:
- `total_segments`: count of scored segments
- `avg_confidence`: mean confidence across all segments
- `supported_pct`, `unsupported_pct`, `uncertain_pct`, `needs_review_pct`: percentage breakdown

---

## Backend API

**Base URL:** `http://127.0.0.1:8000`
**Interactive docs:** `http://127.0.0.1:8000/docs` (Swagger) and `/redoc`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check + model info |
| `POST` | `/predict` | Classify a single article |
| `POST` | `/predict/ensemble` | TF-IDF + gated LIAR transformer |
| `POST` | `/predict/factcheck` | TF-IDF + agentic web-search fact-check |
| `POST` | `/analyze-pdf` | Upload PDF, get per-claim analysis |
| `GET` | `/stats/model-comparison` | Accuracy/F1 for all four models |
| `GET` | `/stats/confusion-matrix` | Confusion matrix for production model |
| `GET` | `/stats/category-breakdown` | Per-subject-category accuracy |
| `GET` | `/stats/trend` | Monthly fake vs. real article volume |
| `GET` | `/limitations` | Full limitations markdown as JSON |

### Request/Response Schemas

#### `POST /predict`

**Request:**
```json
{
  "title": "Fed holds interest rates steady",
  "text": "The Federal Reserve said on Wednesday it would keep interest rates unchanged..."
}
```

**Response (`PredictResponse`):**
```json
{
  "label": "real",
  "confidence": 0.9998,
  "fake_probability": 0.0002,
  "real_probability": 0.9998,
  "model_used": "Linear SVM",
  "disclaimer": "Scores are model-based estimates from a TF-IDF classifier..."
}
```

#### `POST /predict/ensemble`

**Response (`EnsemblePredictResponse`):**
```json
{
  "label": "real",
  "confidence": 0.9998,
  "fake_probability": 0.0002,
  "real_probability": 0.9998,
  "model_used": "Linear SVM",
  "unverifiable": false,
  "liar_signal_used": true,
  "liar_detail": {
    "label": "true",
    "bucket": "leans-real",
    "confidence": 0.8234,
    "model_version": "classifier-v3.0"
  },
  "liar_gate_status": "classifier-v3.0: eval_f1_macro=0.445 clears the trust threshold (0.35)",
  "disclaimer": "Scores are model-based estimates..."
}
```

#### `POST /predict/factcheck`

**Response (`FactCheckResponse`):**
```json
{
  "label": "real",
  "confidence": 0.9998,
  "fake_probability": 0.0002,
  "real_probability": 0.9998,
  "model_used": "Linear SVM",
  "fact_check": {
    "available": true,
    "verdict": "TRUE",
    "confidence": 0.9,
    "explanation": "The claim is supported by multiple sources...",
    "sources": [
      "https://en.wikipedia.org/wiki/Federal_Reserve",
      "https://reuters.com/..."
    ],
    "reason": "ok"
  },
  "disclaimer": "Scores are model-based estimates..."
}
```

#### `POST /analyze-pdf`

**Request:** `multipart/form-data` with a `file` field containing the PDF.

**Response (`AnalyzePdfResponse`):**
```json
{
  "disclaimer": "Scores are model-based estimates...",
  "segments": [
    {
      "segment_text": "The study found a 15% increase in global temperatures over the past decade.",
      "assessment": "supported",
      "confidence_score": 0.8723,
      "explanation": "Text style closely matches credible news writing patterns..."
    },
    {
      "segment_text": "Scientists have proven that the earth is flat.",
      "assessment": "unsupported",
      "confidence_score": 0.8766,
      "explanation": "Text style matches patterns associated with non-credible sources..."
    },
    {
      "segment_text": "The policy may take effect next quarter.",
      "assessment": "uncertain",
      "confidence_score": 0.5201,
      "explanation": "Model cannot strongly associate this text with either credible or non-credible patterns."
    },
    {
      "segment_text": "FYI",
      "assessment": "needs_review",
      "confidence_score": 0.0,
      "explanation": "Segment produced no usable tokens after cleaning."
    }
  ],
  "summary": {
    "total_segments": 42,
    "avg_confidence": 0.72,
    "supported_pct": 61.9,
    "unsupported_pct": 14.3,
    "uncertain_pct": 9.5,
    "needs_review_pct": 14.3
  }
}
```

#### `GET /health`

```json
{
  "status": "ok",
  "model": "Linear SVM",
  "liar_ensemble_gate": "open",
  "liar_ensemble_status": "classifier-v3.0: eval_f1_macro=0.445 clears the trust threshold (0.35)"
}
```

---

## Frontend

The React frontend is a modern single-page application with a dark theme, tabbed interface, and two analysis modes.

### Components

| Component | File | Features |
|-----------|------|----------|
| `Header` | `src/components/Header.jsx` | Title, subtitle ("Document Claim-Support Assessment"), amber eyebrow label |
| `TextInput` | `src/components/TextInput.jsx` | Textarea with mode selector (Classifier / LIAR Ensemble / Fact-Check), results display with probability bars |
| `PdfUploader` | `src/components/PdfUploader.jsx` | Drag-and-drop zone, file validation (PDF, 10MB max), upload progress bar, states: idle/selected/uploading/done/error |
| `ResultsTable` | `src/components/ResultsTable.jsx` | Sortable columns (assessment, confidence), expandable text segments, color-coded rows, confidence bars |
| `SummaryStats` | `src/components/SummaryStats.jsx` | Total segments, average confidence, stacked percentage bar, color legend |
| `FactCheckPanel` | `src/components/FactCheckPanel.jsx` | Verdict badge (TRUE/FALSE/MISLEADING/UNVERIFIED), explanation, clickable source links |
| `EnsemblePanel` | `src/components/EnsemblePanel.jsx` | LIAR bucket display (leans-real/leans-fake/uncertain), 6-class label, gate status |
| `App` | `src/App.jsx` | Tab management, state orchestration, footer with tech stack details |

### UI Features

- **Two analysis tabs:**
  - **Text Analysis**: paste text, choose mode (classifier only / + LIAR ensemble / + web fact-check), see results inline
  - **PDF Upload**: drag-and-drop upload with progress, per-claim results table, summary statistics
- **Three prediction modes** (text analysis tab):
  - Classifier only: fast TF-IDF prediction via `/predict`
  - + LIAR ensemble: adds RoBERTa style signal via `/predict/ensemble`
  - + Web fact-check: adds Wikipedia + Groq verdict via `/predict/factcheck`
- **Drag-and-drop PDF upload** with visual feedback (border color change)
- **File validation** — PDF only, max 10MB, clear error messages
- **Upload progress bar** via axios `onUploadProgress`
- **Color-coded results table** — green (supported), red (unsupported), yellow (uncertain), gray (needs review)
- **Sortable columns** — click header to sort by assessment or confidence
- **Expandable text segments** — long claims truncated with show/hide toggle
- **Summary statistics** — stacked bar chart of assessment distribution
- **Responsive design** — works on mobile and desktop
- **Keyboard shortcut** — Ctrl+Enter to run text analysis
- **Dark theme** — consistent with the existing dashboard's visual language

### Styling & Theme

Custom CSS properties defined in `src/index.css`:

```css
--bg: #10161d          /* Background */
--panel: #1a2129       /* Panel background */
--panel-2: #212b34     /* Secondary panel */
--border: #2b3540      /* Border color */
--text-primary: #edeae0 /* Primary text */
--text-secondary: #a6a395 /* Secondary text */
--text-muted: #6e6c60  /* Muted text */
--real: #4fb3a0        /* Real/supported (teal) */
--fake: #e2604f        /* Fake/unsupported (coral) */
--amber: #e8a83d       /* Uncertain (amber) */
```

Typography: Space Grotesk (body), Newsreader (headings), IBM Plex Mono (labels/data).

---

## Legacy Dashboard

A self-contained HTML dashboard in `dashboard/index.html` that embeds all reports and a compact client-side model. Built from `dashboard/template.html` by `src/build_dashboard.py`.

### Features

- **Live prediction desk**: paste text, get instant verdict using either the production API or the bundled in-browser fallback model
- **Model bench chart**: bar chart comparing all four classifiers (Chart.js)
- **Confusion matrix**: production model error analysis
- **Category breakdown**: per-subject accuracy bars
- **Trend chart**: monthly fake vs. real volume over time
- **Ablation table**: dateline leakage before/after comparison
- **Fallback model**: automatically falls back to 3,500-term Logistic Regression when API is unreachable

### API Base Configuration

The dashboard resolves the API base URL in this priority order:
1. `?api=https://your-api.example.com` (query parameter)
2. `window.FND_API_BASE` (global variable)
3. `http://127.0.0.1:8000` (default)

---

## Setup

### Prerequisites

- **Python 3.9+**
- **Node.js 18+** (for frontend)
- **Kaggle dataset**: `Fake.csv` and `True.csv` from [Kaggle's Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon

### Backend Setup

```bash
# Clone the repository
git clone <repository-url>
cd Fake-News-Detection1

# Create virtual environment
python -m venv myenv

# Activate (Linux/macOS)
source myenv/bin/activate

# Activate (Windows)
myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Dataset

Place the following files in `data/`:
- `data/Fake.csv` — columns: `title, text, subject, date`
- `data/True.csv` — columns: `title, text, subject, date`

**Note:** The dataset is not bundled due to file size (~110 MB) and licensing. Download directly from Kaggle.

### Training Models

After placing the dataset, train the models:

```bash
# Train all classifiers and generate reports
python src/train.py

# Run dateline leakage ablation study
python src/ablation_dateline.py

# Export compact client-side model
python src/export_js_model.py

# Build legacy dashboard (optional)
python src/build_dashboard.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE` | `/api` | Backend API base URL for the frontend |
| `GROQ_API_KEY` | *(none)* | Groq API key for agentic fact-check (optional) |

**Setting `GROQ_API_KEY`:** nothing in this codebase reads the `key` file automatically —
`src/fact_check.py` only reads `GROQ_API_KEY` from the process environment
(`os.environ.get("GROQ_API_KEY")`). Export it yourself before starting the backend, e.g.:

```bash
# from the project root, using the local `key` file
export GROQ_API_KEY=$(cat key)
uvicorn api.main:app --reload --port 8000
```

or set it in a `.env` file and load it with `python-dotenv` / your process manager of choice.
`GET /health` and the `[startup]` log line report whether the fact-checker actually picked
the key up (`READY` vs `DISABLED`).

**Security note:** the `key` file is excluded via `.gitignore` and should never be committed.
Treat any Groq key that has left your local machine (e.g. included in a zipped project export,
pasted into a chat, or shared with a third party) as compromised and rotate it from the
[Groq console](https://console.groq.com/keys), since anyone who has seen it can spend against
your account.

**Model availability:** Groq periodically deprecates models (see their
[deprecations page](https://console.groq.com/docs/deprecations)). `src/fact_check.py` currently
targets `openai/gpt-oss-120b` for generation and `groq/compound-mini` (restricted to its
`web_search` tool) for retrieval — if fact-check calls start failing with 404s again, check
that page first.

---

## Running

### Full Stack (Recommended)

**Terminal 1 — Backend:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser. The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

### Backend Only

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://127.0.0.1:8000/docs`.

### Frontend Only (with Remote API)

```bash
cd frontend
VITE_API_BASE=https://your-api.example.com npm run dev
```

### Legacy Dashboard

Open `dashboard/index.html` directly in a browser. It works standalone with an optional API backend.

---

## Model Performance

### Held-out Test Set (8,978 articles, 35,911 training)

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | Precision | Recall | Inference Time |
|-------|----------|---------------|------------|-----------|--------|----------------|
| **Linear SVM (production)** | **0.9937** | **0.9937** | **0.9936** | 0.9937 | 0.9937 | 0.016s |
| Random Forest | 0.9893 | 0.9893 | 0.9893 | 0.9893 | 0.9893 | 1.433s |
| Logistic Regression | 0.9881 | 0.9881 | 0.9881 | 0.9881 | 0.9882 | 0.005s |
| Multinomial Naive Bayes | 0.9581 | 0.9581 | 0.9580 | 0.9581 | 0.9581 | 0.008s |

All four models clear the MVP target (F1 >= 0.90). Linear SVM was selected based on weighted F1. Full metrics in `reports/model_comparison.json`.

### Confusion Matrix (Linear SVM)

|  | Predicted Fake | Predicted Real |
|--|---------------|---------------|
| **Actual Fake** | 4,665 (correct) | 30 (missed) |
| **Actual Real** | 27 (wrongly flagged) | 4,256 (correct) |

- **False negatives** (fake labeled as real): 30 (0.33%)
- **False positives** (real labeled as fake): 27 (0.30%)
- Total errors: 57 out of 8,978 (0.63%)

### Per-Category Accuracy

| Subject | Articles | Fake | Real | Accuracy |
|---------|----------|------|------|----------|
| politicsNews | 2,195 | 0 | 2,195 | 98.95% |
| worldnews | 2,088 | 0 | 2,088 | 99.81% |
| News | 1,771 | 1,771 | 0 | 99.94% |
| politics | 1,353 | 1,353 | 0 | 98.97% |
| left-news | 906 | 906 | 0 | 99.56% |
| Government News | 318 | 318 | 0 | 96.54% |
| Middle-east | 174 | 174 | 0 | 100.00% |
| US_News | 173 | 173 | 0 | 100.00% |

**Important caveat:** Each `subject` tag belongs to only one class in this corpus (politicsNews/worldnews = 100% real, all others = 100% fake). This is a dataset artifact, not a real-world signal. The model does not use `subject` as an input feature.

### Corpus Trend Data

Monthly article volume from 2015–2018:

| Period | Fake Articles | Real Articles | Notes |
|--------|--------------|---------------|-------|
| 2015 | ~2,835 | 0 | Only fake articles in corpus |
| 2016 Q1-Q4 | ~10,842 | ~4,335 | Peak fake volume (~1,000/month) |
| 2017 Q1-Q3 | ~6,693 | ~4,420 | Declining fake, growing real |
| 2017 Q4 | ~1,575 | ~8,801 | Real articles surge (~3,000/month) |

---

## Known Limitations

1. **Narrow dataset** — Real articles are almost entirely Reuters wire copy; fake articles from a small set of outlets. Skews heavily to political/world news. Has not seen sports, entertainment, science, health, or local news.

2. **Subject tag is a perfect class proxy** — `politicsNews`/`worldnews` are 100% real; other tags are 100% fake. This is a dataset artifact, not a real-world signal. The model does not use `subject` as input.

3. **Residual source-style bias** — After stripping datelines, Reuters house style (formal attribution, "said", weekday-dated reporting) still influences predictions. Part of what the model learned is "this reads like wire journalism" rather than "this is factually accurate."

4. **Class balance** — Training corpus is ~50/50 fake/real. Real-world traffic will differ. Decision threshold should be re-tuned for deployment.

5. **Style mimicry** — Advertisers deliberately imitating wire-service formatting will erode accuracy. The model should be retrained periodically on fresh, labeled examples.

6. **LIAR model quality** — The bundled RoBERTa transformer (eval_f1_macro=0.445 on 6 classes) is gated and may be disabled if a worse model is dropped in. The gate re-evaluates automatically against the new manifest.

7. **Fact-check dependency** — The agentic fact-checker requires a valid `GROQ_API_KEY` and internet access. Without it, the endpoint gracefully falls back to classifier-only results.

8. **PDF limitations** — Scanned/image-based PDFs will produce no extractable text. The API returns a clear error in this case.

---

## Responsible Use

- Treat predictions as a **triage signal for human fact-checkers**, not an automated takedown mechanism
- The model answers "does this resemble credible or non-credible news **style**" — NOT "is this claim true"
- Log prediction confidence alongside outcomes so drift can be monitored
- Re-validate on a sample of the actual target domain before trusting reported metrics
- Retrain periodically to counter style mimicry and data drift
- Full responsible-use note: `docs/limitations.md`

---

## Project Structure Reference

For a complete file-by-file reference, see [`docs/DIRECTORY_STRUCTURE.md`](docs/DIRECTORY_STRUCTURE.md).