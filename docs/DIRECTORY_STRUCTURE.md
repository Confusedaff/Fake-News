# Directory Structure

Complete file-by-file reference for the Fake News Detection project.

---

## Root

| File | Description |
|------|-------------|
| `README.md` | Project documentation: architecture, setup, API reference, model performance |
| `requirements.txt` | Python dependencies (FastAPI, scikit-learn, PyPDF2, etc.) |
| `.gitignore` | Git exclusions (`myenv/` virtual environment) |

---

## `api/` — Backend API

| File | Description |
|------|-------------|
| `api/main.py` | FastAPI serving application. Defines all REST endpoints (`/health`, `/predict`, `/predict/ensemble`, `/analyze-pdf`, `/stats/*`, `/limitations`). Loads models at startup, handles CORS, request validation via Pydantic, and error responses. |
| `api/pdf_processor.py` | PDF processing pipeline. `extract_text_from_pdf()` uses PyPDF2 to pull text from all pages. `split_into_claims()` splits on sentence boundaries with line-based fallback. `score_segments()` runs each claim through the TF-IDF + classifier pipeline. `compute_summary()` calculates aggregate statistics. |

---

## `frontend/` — React Frontend

| File | Description |
|------|-------------|
| `frontend/package.json` | Node.js project config. Dependencies: React 18, Axios, Lucide React. Dev deps: Vite 6, Tailwind CSS 4. |
| `frontend/vite.config.js` | Vite build configuration. React + Tailwind plugins. Dev server on port 5173 with proxy: `/api` -> `http://127.0.0.1:8000`. |
| `frontend/index.html` | HTML entry point. Loads Google Fonts (Newsreader, Space Grotesk, IBM Plex Mono). |
| `frontend/.gitignore` | Excludes `node_modules/`, `dist/`, `.env` files. |
| `frontend/src/main.jsx` | React entry point. Mounts `<App />` into `#root` with StrictMode. |
| `frontend/src/App.jsx` | Main application component. Manages state for results and errors. Renders Header, PdfUploader, SummaryStats, ResultsTable, Disclaimer, and footer. |
| `frontend/src/api.js` | Axios instance configured with base URL (defaults to `/api`, configurable via `VITE_API_BASE` env var) and 60s timeout. |
| `frontend/src/index.css` | Global styles. Tailwind import. CSS custom properties for the dark theme color palette (bg, panels, borders, semantic colors). Font family definitions. |
| `frontend/src/components/Header.jsx` | App header with title, subtitle, model statistics (name, features, F1), and a visible disclaimer banner with ShieldAlert icon. |
| `frontend/src/components/PdfUploader.jsx` | PDF upload component. Drag-and-drop zone with visual feedback. Hidden file input for click-to-browse. File validation (PDF only, 10MB max). Upload progress bar. States: idle, selected, uploading, done, error. Cancel and reset functionality. |
| `frontend/src/components/ResultsTable.jsx` | Sortable data table for per-claim results. Columns: row number, text segment (expandable), assessment (color-coded badge), confidence (inline progress bar), explanation. Sort by assessment or confidence. Mobile-responsive with hidden columns on small screens. |
| `frontend/src/components/SummaryStats.jsx` | Aggregate statistics display. Shows total segments and average confidence as large numbers. Stacked horizontal bar showing percentage breakdown of all four assessment categories. Color legend below the bar. |
| `frontend/src/components/Disclaimer.jsx` | Responsible-use notice. Explains the model's limitations: narrow training data, pattern-based scoring (not truth verification), style mimicry risks, and recommended triage-only use. Styled with amber border matching the existing dashboard's "editor's note" aesthetic. |

---

## `src/` — Training Pipeline & Shared Code

| File | Description |
|------|-------------|
| `src/preprocessing.py` | Shared text cleaning and dataset loading. `clean_text()` lowercases, strips Reuters datelines, removes URLs/punctuation/stopwords, collapses whitespace. Used by both training and serving to prevent train/serve skew. `build_dataset()` loads and merges the Kaggle CSVs. Contains the compact English stopword list and regex patterns for datelines, URLs, and non-alpha characters. |
| `src/train.py` | Trains and evaluates four classifiers (Logistic Regression, Multinomial NB, Linear SVM, Random Forest) on TF-IDF features. Selects the best by weighted F1. Saves the winning model + vectorizer to `models/` and all metrics to `reports/`. Uses fixed random seed (42) for reproducibility. |
| `src/liar_ensemble.py` | Gated ensemble support. `LiarEnsembleModel` wraps a fine-tuned RoBERTa-base (classifier-v3.0) for 6-class truthfulness classification. Evaluates a quality gate (`MIN_F1_MACRO_TO_TRUST = 0.35`) against the model's own reported validation F1. Loads the transformer lazily (only if gate passes and prediction requested). Module-level singleton `liar_model` evaluates the gate at import time. |
| `src/export_js_model.py` | Exports a compact 3,500-term Logistic Regression model as JSON (`reports/js_model.json`) for the in-browser fallback demo. Trades accuracy for portability — the exported model is small enough to embed inline in HTML. |
| `src/build_dashboard.py` | Generates `dashboard/index.html` from `dashboard/template.html` by injecting all `reports/*.json` data via the `__EMBEDDED_DATA__` placeholder. Must run after `train.py` and `export_js_model.py`. |
| `src/ablation_dateline.py` | Measures the dateline-leakage artifact by training an identical pipeline twice (with and without the Reuters dateline). Outputs `reports/dateline_leakage_ablation.json` with before/after accuracy, F1, and top terms. |

---

## `models/` — Serialized Models & Artifacts

| File | Description |
|------|-------------|
| `models/best_model.joblib` | Serialized production model (Linear SVM wrapped in `CalibratedClassifierCV`). |
| `models/vectorizer.joblib` | Serialized fitted `TfidfVectorizer` (30,000 features, unigrams + bigrams, sublinear TF). |
| `models/best_model_name.txt` | Plain-text name of the selected model: `Linear SVM`. |

### `models/liar_v3/` — LIAR Transformer Model

| File | Description |
|------|-------------|
| `models/liar_v3/manifest.json` | Model card: classifier-v3.0, roberta-base, eval_f1_macro=0.445, dataset=LIAR. |
| `models/liar_v3/metadata.json` | Model metadata: status=READY, max_length=256, dataset info. |
| `models/liar_v3/calibration.json` | Temperature scaling config: T=1.227, ECE before/after calibration. |
| `models/liar_v3/label_mapping.json` | 6-class label-to-ID mapping (0=pants-fire ... 5=true). |
| `models/liar_v3/labels.json` | Bidirectional label <-> ID mapping. |
| `models/liar_v3/model/config.json` | RoBERTa architecture: 12 layers, 768 hidden, 12 heads, vocab 50265. |
| `models/liar_v3/model/model.safetensors` | Fine-tuned model weights. |
| `models/liar_v3/model/tokenizer.json` | Tokenizer configuration. |
| `models/liar_v3/model/tokenizer.model` | SentencePiece tokenizer model. |
| `models/liar_v3/model/tokenizer_config.json` | Tokenizer settings. |
| `models/liar_v3/model/vocab.json` | Vocabulary file. |
| `models/liar_v3/model/merges.txt` | BPE merge rules. |
| `models/liar_v3/model/special_tokens_map.json` | Special token definitions. |
| `models/liar_v3/model/training_args.bin` | Training arguments used for fine-tuning. |
| `models/liar_v3/tokenizer/` | Separate tokenizer directory (same files as model/tokenizer/). |

---

## `reports/` — Metrics & Exported Artifacts

| File | Description |
|------|-------------|
| `reports/model_comparison.json` | Accuracy, precision, recall, F1 (weighted + macro), inference time for all four models. Also contains `best_model`, `n_train`, `n_test`, `n_features`. |
| `reports/confusion_matrices.json` | 2x2 confusion matrix for each model (predicted fake/real vs actual fake/real). |
| `reports/category_breakdown.json` | Per-subject-category accuracy (e.g. `politicsNews`, `worldnews`). |
| `reports/trend_data.json` | Monthly fake vs. real article counts from 2015-03 to 2018-02. |
| `reports/dateline_leakage_ablation.json` | Before/after comparison: accuracy, F1, and top terms when Reuters dateline is left in vs stripped. |
| `reports/js_model.json` | Compact 3,500-term Logistic Regression weights exported for the in-browser fallback demo. Contains intercept, term->\[idf, coefficient\] mapping, and metadata. |

---

## `dashboard/` — Legacy Static Dashboard

| File | Description |
|------|-------------|
| `dashboard/template.html` | Source template for the static dashboard. Contains `__EMBEDDED_DATA__` placeholder that `src/build_dashboard.py` replaces with report data. Edit this file, not `index.html`. |
| `dashboard/index.html` | Generated dashboard. Self-contained HTML with embedded Chart.js charts, inline JS TF-IDF model, and live API demo. Do not edit directly — will be overwritten by `build_dashboard.py`. |

---

## `data/` — Dataset

| File | Description |
|------|-------------|
| `data/README.md` | Instructions to download `Fake.csv` and `True.csv` from Kaggle's "Fake and Real News Dataset" (by Clement Bisaillon). Not bundled due to file size (~110 MB) and licensing. |

---

## `docs/` — Documentation

| File | Description |
|------|-------------|
| `docs/limitations.md` | Limitations and responsible-use note (required MVP deliverable). Covers narrow dataset, subject-tag proxy, dateline leakage, class imbalance, style mimicry, and recommended use as triage signal. |
| `docs/DIRECTORY_STRUCTURE.md` | This file — complete file-by-file directory reference. |
