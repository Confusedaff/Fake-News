"""
Serving API for the Fake News Detection MVP (Objective 4, item 4: "Deploy
the best-performing model behind an API that returns a prediction with a
confidence score.")

Run with:  uvicorn api.main:app --reload --port 8000   (from the project root)

Endpoints
---------
GET  /health                      liveness check
POST /predict                     {title?, text} -> label, confidence, category
POST /predict/ensemble            {title?, text} -> TF-IDF verdict + gated LIAR "unverifiable" flag
POST /predict/factcheck            {title?, text} -> TF-IDF verdict + agentic web-search fact-check verdict
POST /predict/combined              {title?, text} -> ONE final verdict+confidence combining all 3 signals, plus each signal individually
POST /analyze-pdf                  file, ?fact_check=false, ?max_fact_checks=8 -> per-segment TF-IDF scoring, optionally + fact-check on unsupported/uncertain segments
GET  /stats/model-comparison       accuracy/F1 per benchmarked model
GET  /stats/confusion-matrix       confusion matrix for the production model
GET  /stats/category-breakdown     accuracy by article subject
GET  /stats/trend                  monthly fake vs. real volume in the training corpus
GET  /limitations                  the stated limitations note
"""
import json
import sys
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import clean_text  # noqa: E402
from liar_ensemble import liar_model  # noqa: E402
from fact_check import fact_checker  # noqa: E402
from ensemble_combine import combine, decide_web_search_trigger  # noqa: E402
from api.pdf_processor import (
    extract_text_from_pdf,
    split_into_claims,
    score_segments,
    compute_summary,
    enrich_with_fact_check,
)

app = FastAPI(
    title="Fake News Detection API",
    description="Use Case #15 — GenAI & Security. Serves TF-IDF + classical "
                "classifier predictions with confidence scores, plus the "
                "analytics that back the dashboard.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DISCLAIMER = (
    "Scores are model-based estimates from a TF-IDF classifier trained on "
    "a narrow news article dataset. They indicate pattern similarity to "
    "known credible or non-credible sources, NOT definitive truth. "
    "Use as a triage signal for human review, not as an automated verdict."
)

vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
model = joblib.load(MODELS_DIR / "best_model.joblib")
best_model_name = (MODELS_DIR / "best_model_name.txt").read_text().strip()

print(f"[startup] LIAR ensemble gate: {'OPEN' if liar_model.trusted else 'CLOSED'} -- {liar_model.reason}")
print(f"[startup] Agentic fact-check: {'READY' if fact_checker.available else 'DISABLED'} -- {fact_checker.reason}")


class PredictRequest(BaseModel):
    title: str = Field("", description="Article headline (optional)")
    text: str = Field(..., description="Article body text", min_length=1)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    model_used: str
    disclaimer: str = Field(default=DISCLAIMER, description="Responsible-use disclaimer")


class EnsemblePredictResponse(BaseModel):
    label: str                       # from TF-IDF, unchanged -- LIAR never overrides this
    confidence: float
    fake_probability: float
    real_probability: float
    model_used: str
    unverifiable: bool                # True only when the LIAR signal is trusted AND lands in the middle buckets
    liar_signal_used: bool            # whether the LIAR model contributed anything to this response at all
    liar_detail: Optional[dict] = None  # 6-class label/bucket/confidence when liar_signal_used is True
    liar_gate_status: str             # human-readable reason the gate is open or closed, for transparency
    disclaimer: str = Field(default=DISCLAIMER, description="Responsible-use disclaimer")


class FactCheckDetail(BaseModel):
    available: bool
    verdict: Optional[str] = None          # TRUE | FALSE | MISLEADING | UNVERIFIED
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    reason: Optional[str] = None           # "ok", or why it's unavailable


class FactCheckResponse(BaseModel):
    # Classifier result, unchanged in meaning from /predict -- the agentic
    # fact-check is an ADDITIONAL signal, never a replacement or override.
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    model_used: str
    fact_check: FactCheckDetail
    disclaimer: str = Field(default=DISCLAIMER, description="Responsible-use disclaimer")


class TfidfBlock(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    model_used: str


class LiarBlock(BaseModel):
    label: str                # 6-class LIAR label
    bucket: str                # leans-real | leans-fake | uncertain
    direction: str             # same 3 values, named for direct comparison with tfidf.label
    confidence: float
    model_version: Optional[str] = None


class CombinedPredictResponse(BaseModel):
    # --- the one final answer -------------------------------------------
    final_label: str                    # real | fake | uncertain
    final_confidence: float
    final_source: str                   # "weighted_blend" -- see weights_used for the actual mix
    explanation: str
    weights_used: dict[str, float] = Field(
        default_factory=dict,
        description="Normalized weight each available signal contributed to the blend, e.g. {'web': 0.6, 'tfidf': 0.25, 'liar': 0.15}",
    )

    # --- every individual signal, always shown in full -------------------
    tfidf: TfidfBlock
    liar: Optional[LiarBlock] = None
    liar_gate_status: str
    fact_check: Optional[FactCheckDetail] = None

    # --- transparency into the process ------------------------------------
    web_search_triggered: bool
    web_search_trigger_reason: str

    disclaimer: str = Field(default=DISCLAIMER, description="Responsible-use disclaimer")


class SegmentResult(BaseModel):
    segment_text: str
    assessment: str
    confidence_score: float
    explanation: str
    # Populated only when the caller opts into ?fact_check=true; None means
    # "not attempted" (either fact-check wasn't requested, or this segment
    # wasn't in the capped/targeted subset that got checked) -- distinct
    # from an attempted-but-unavailable result, which fills this in with
    # available=False and a reason.
    fact_check: Optional[FactCheckDetail] = None


class AnalysisSummary(BaseModel):
    total_segments: int
    avg_confidence: float
    supported_pct: float
    unsupported_pct: float
    uncertain_pct: float
    needs_review_pct: float


class AnalyzePdfResponse(BaseModel):
    disclaimer: str
    segments: list[SegmentResult]
    summary: AnalysisSummary


def _load_report(name: str):
    path = REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Report {name} not generated yet. Run src/train.py.")
    return json.loads(path.read_text())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": best_model_name,
        "liar_ensemble_gate": "open" if liar_model.trusted else "closed",
        "liar_ensemble_status": liar_model.reason,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    content = f"{req.title}. {req.text}" if req.title else req.text
    cleaned = clean_text(content, remove_dateline=True)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Input text produced no usable tokens after cleaning.")
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]  # [P(fake), P(real)]
    fake_p, real_p = float(proba[0]), float(proba[1])
    label = "real" if real_p >= fake_p else "fake"
    confidence = max(fake_p, real_p)
    return PredictResponse(
        label=label, confidence=round(confidence, 4),
        fake_probability=round(fake_p, 4), real_probability=round(real_p, 4),
        model_used=best_model_name,
        disclaimer=DISCLAIMER,
    )


@app.post("/predict/ensemble", response_model=EnsemblePredictResponse)
def predict_ensemble(req: PredictRequest):
    """
    TF-IDF remains the sole source of the fake/real label and confidence --
    it's the model with an actual validated 0.99+ track record on this
    task. The LIAR model (classifier-v3.0) only ever *adds* an
    `unverifiable` flag on top, and only if its own reported validation
    F1 clears MIN_F1_MACRO_TO_TRUST in src/liar_ensemble.py. Today that
    gate is open (bundled v3.0 scores eval_f1_macro=0.445, which clears
    the 0.35 threshold), so liar_signal_used will be True and unverifiable
    reflects the LIAR signal. Re-check liar_model.trusted / GET /health if
    a future checkpoint's eval_f1_macro drops below the threshold.
    """
    content = f"{req.title}. {req.text}" if req.title else req.text
    cleaned = clean_text(content, remove_dateline=True)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Input text produced no usable tokens after cleaning.")

    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    fake_p, real_p = float(proba[0]), float(proba[1])
    label = "real" if real_p >= fake_p else "fake"
    confidence = max(fake_p, real_p)

    liar_detail = None
    unverifiable = False
    try:
        liar_detail = liar_model.liar_signal(content)
    except Exception as exc:  # model files present but fail to load/run -- fail safe, not 500
        liar_detail = None
        liar_gate_note = f"LIAR signal errored at inference time ({exc}); falling back to TF-IDF-only"
    else:
        liar_gate_note = liar_model.reason

    if liar_detail is not None:
        unverifiable = liar_detail["bucket"] == "uncertain"

    return EnsemblePredictResponse(
        label=label, confidence=round(confidence, 4),
        fake_probability=round(fake_p, 4), real_probability=round(real_p, 4),
        model_used=best_model_name,
        unverifiable=unverifiable,
        liar_signal_used=liar_detail is not None,
        liar_detail=liar_detail,
        liar_gate_status=liar_gate_note,
    )


@app.post("/predict/factcheck", response_model=FactCheckResponse)
def predict_factcheck(req: PredictRequest):
    """
    Adds a real-world, evidence-grounded verdict on top of the TF-IDF
    style classifier. The classifier answers "does this resemble credible
    or non-credible news style"; this answers "is this claim actually true
    right now, per live web search" -- a different question the classifier
    structurally cannot answer (see docs/limitations.md).

    The TF-IDF label/confidence are computed and returned exactly as in
    /predict and are never overwritten by the fact-check result. If the
    fact-check call is unavailable or fails for any reason, fact_check.available
    is False with a human-readable reason, and the classifier fields are
    still valid -- this endpoint never 500s because the LLM call failed.
    """
    content = f"{req.title}. {req.text}" if req.title else req.text
    cleaned = clean_text(content, remove_dateline=True)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Input text produced no usable tokens after cleaning.")

    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    fake_p, real_p = float(proba[0]), float(proba[1])
    label = "real" if real_p >= fake_p else "fake"
    confidence = max(fake_p, real_p)

    fc = fact_checker.check(content)

    return FactCheckResponse(
        label=label, confidence=round(confidence, 4),
        fake_probability=round(fake_p, 4), real_probability=round(real_p, 4),
        model_used=best_model_name,
        fact_check=FactCheckDetail(
            available=fc.available,
            verdict=fc.verdict,
            confidence=fc.confidence,
            explanation=fc.explanation,
            sources=fc.sources,
            reason=fc.reason,
        ),
    )


@app.post("/predict/combined", response_model=CombinedPredictResponse)
def predict_combined(req: PredictRequest):
    """
    Runs all three signals -- TF-IDF classifier, gated RoBERTa/LIAR style
    signal, and agentic web-search fact-check -- and combines them into
    ONE final label + confidence via a weighted blend (web fact-check
    weighted highest at 60%, TF-IDF 25%, LIAR 15% when all three are
    available -- see src/ensemble_combine.py for the exact rule and
    renormalization when a signal is unavailable), while still returning
    every individual signal in full so the caller can see exactly which
    model said what and how much it counted.

    The web fact-check always runs, independent of what TF-IDF/LIAR say
    -- every request is checked against live, real-world evidence.
    web_search_triggered / web_search_trigger_reason are kept in the
    response so the frontend still has something to display, but
    web_search_triggered will always be True.
    """
    content = f"{req.title}. {req.text}" if req.title else req.text
    cleaned = clean_text(content, remove_dateline=True)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Input text produced no usable tokens after cleaning.")

    # --- signal 1: TF-IDF classifier (fast, local) ---
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    fake_p, real_p = float(proba[0]), float(proba[1])
    tfidf_label = "real" if real_p >= fake_p else "fake"
    tfidf_confidence = max(fake_p, real_p)

    # --- signal 2: gated LIAR/RoBERTa style signal (fast, local) ---
    liar_detail = None
    try:
        liar_detail = liar_model.liar_signal(content)
    except Exception as exc:  # fail safe, never 500 on a bundled-model issue
        liar_detail = None
        liar_gate_note = f"LIAR signal errored at inference time ({exc}); falling back to TF-IDF-only"
    else:
        liar_gate_note = liar_model.reason

    # --- signal 3: web fact-check (now unconditional -- always runs) ---
    should_search, trigger_reason = decide_web_search_trigger(tfidf_label, tfidf_confidence, liar_detail)
    fact_check_result = fact_checker.check(content) if should_search else None

    result = combine(
        tfidf_label=tfidf_label,
        tfidf_confidence=tfidf_confidence,
        fake_probability=fake_p,
        real_probability=real_p,
        model_used=best_model_name,
        liar_detail=liar_detail,
        liar_gate_status=liar_gate_note,
        fact_check_result=fact_check_result,
        web_search_triggered=should_search,
        web_search_trigger_reason=trigger_reason,
    )

    return CombinedPredictResponse(
        final_label=result.final_label,
        final_confidence=result.final_confidence,
        final_source=result.final_source,
        explanation=result.explanation,
        weights_used=result.weights_used,
        tfidf=TfidfBlock(**result.tfidf),
        liar=LiarBlock(**result.liar) if result.liar else None,
        liar_gate_status=liar_gate_note,
        fact_check=FactCheckDetail(**result.fact_check) if result.fact_check else None,
        web_search_triggered=result.web_search_triggered,
        web_search_trigger_reason=result.web_search_trigger_reason,
    )


@app.post("/analyze-pdf", response_model=AnalyzePdfResponse)
async def analyze_pdf(
    file: UploadFile = File(...),
    fact_check: bool = False,
    max_fact_checks: int = 8,
):
    """Accept a PDF upload, extract text, split into claims, and score each.

    The TF-IDF classifier score (assessment/confidence/explanation) always
    runs -- it answers "does this text's style resemble credible/non-credible
    news sources," which is NOT the same question as "is this true" (see
    docs/limitations.md). Pass ?fact_check=true to additionally layer a
    real-world, evidence-grounded verdict (with source URLs) from
    src/fact_check.py onto a capped, targeted subset of segments -- see
    enrich_with_fact_check() in api/pdf_processor.py for why it's capped
    and which segments get prioritized. ?max_fact_checks=N overrides the
    default cap of 8.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text = extract_text_from_pdf(contents)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the PDF. The file may be scanned/image-based.",
        )

    segments = split_into_claims(text)
    if not segments:
        raise HTTPException(
            status_code=400,
            detail="No valid text segments found after splitting the extracted content.",
        )

    results = score_segments(segments, vectorizer, model, clean_text)

    if fact_check:
        results = enrich_with_fact_check(results, fact_checker, max_checks=max_fact_checks)
    else:
        for r in results:
            r["fact_check"] = None

    summary = compute_summary(results)

    return AnalyzePdfResponse(
        disclaimer=DISCLAIMER,
        segments=[SegmentResult(**r) for r in results],
        summary=AnalysisSummary(**summary),
    )


@app.get("/stats/model-comparison")
def model_comparison():
    return _load_report("model_comparison.json")


@app.get("/stats/confusion-matrix")
def confusion_matrix():
    data = _load_report("confusion_matrices.json")
    return data.get(best_model_name, data)


@app.get("/stats/category-breakdown")
def category_breakdown():
    return _load_report("category_breakdown.json")


@app.get("/stats/trend")
def trend():
    return _load_report("trend_data.json")


@app.get("/limitations")
def limitations():
    path = ROOT / "docs" / "limitations.md"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Limitations note not generated yet.")
    return {"limitations_markdown": path.read_text()}