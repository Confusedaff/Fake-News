"""
PDF text extraction, claim splitting, and per-claim scoring for the
Document Claim-Support Assessment tool.

Pipeline: PDF bytes -> raw text -> sentence-level segments -> each segment
scored by the production TF-IDF classifier -> assessment label + confidence.
"""
import re
from io import BytesIO
from statistics import mean

from PyPDF2 import PdfReader

# Sentence boundary: period/exclamation/question followed by whitespace.
# Keeps abbreviations like "U.S." together by requiring a space after.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Segments shorter than this (after cleaning) are too noisy to score.
MIN_SEGMENT_CHARS = 10


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF, page by page."""
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def split_into_claims(text: str) -> list[str]:
    """Split raw text into sentence-level segments.

    Falls back to line-based splitting if the text has no sentence
    punctuation (e.g. bullet lists, fragment-heavy documents).
    """
    text = text.strip()
    if not text:
        return []

    # Try sentence splitting first.
    sentences = SENTENCE_RE.split(text)

    # If we only got one giant chunk, there are no sentence endings --
    # fall back to line-based splitting.
    if len(sentences) <= 1:
        sentences = [line.strip() for line in text.splitlines() if line.strip()]

    # Filter out noise: very short fragments, page numbers, headers.
    segments = []
    for s in sentences:
        s = s.strip()
        if len(s) >= MIN_SEGMENT_CHARS:
            segments.append(s)

    return segments


def score_segments(
    segments: list[str],
    vectorizer,
    model,
    clean_text_fn,
) -> list[dict]:
    """Score each segment using the production TF-IDF + classifier.

    Returns a list of dicts with keys:
        segment_text, assessment, confidence_score, explanation
    """
    results = []

    for raw_segment in segments:
        cleaned = clean_text_fn(raw_segment, remove_dateline=True)

        if not cleaned.strip():
            results.append({
                "segment_text": raw_segment,
                "assessment": "needs_review",
                "confidence_score": 0.0,
                "explanation": "Segment produced no usable tokens after cleaning (too short, all stopwords, or non-English).",
            })
            continue

        X = vectorizer.transform([cleaned])
        proba = model.predict_proba(X)[0]
        fake_p, real_p = float(proba[0]), float(proba[1])

        if real_p >= 0.85:
            assessment = "supported"
            confidence = real_p
            explanation = (
                f"Text style closely matches credible news writing patterns "
                f"(formal attribution, structured reporting)."
            )
        elif fake_p >= 0.85:
            assessment = "unsupported"
            confidence = fake_p
            explanation = (
                f"Text style matches patterns associated with non-credible sources "
                f"in the training data."
            )
        else:
            assessment = "uncertain"
            confidence = max(fake_p, real_p)
            explanation = (
                f"Model cannot strongly associate this text with either credible or "
                f"non-credible news patterns. This may be non-news content (emails, "
                f"instructions, etc.) which the model was not trained to assess."
            )

        results.append({
            "segment_text": raw_segment,
            "assessment": assessment,
            "confidence_score": round(confidence, 4),
            "explanation": explanation,
        })

    return results


def enrich_with_fact_check(
    results: list[dict],
    fact_checker,
    target_assessments: tuple = ("unsupported", "uncertain"),
    max_checks: int = 8,
) -> list[dict]:
    """
    Layers a real-world, evidence-grounded verdict (with source URLs) onto
    a capped subset of already-scored segments. Mutates and returns the
    same list -- every result dict gains a "fact_check" key.

    Why capped and targeted rather than run on every segment
    ----------------------------------------------------------
    - Cost/latency: each fact-check call is 2-3 HTTP round trips
      (Wikipedia + a Groq web-search call + a Groq generation call). A
      multi-page PDF can produce 50+ segments; fact-checking all of them
      on every upload would be slow and could trip Groq's per-minute rate
      limits, especially on the free tier.
    - Where it adds the most value: the TF-IDF classifier is a *style*
      classifier (see docs/limitations.md) -- it answers "does this read
      like known credible/non-credible news formatting," not "is this
      true." Segments it already flagged "unsupported" or "uncertain" are
      exactly the ones where that style-guess is most likely to diverge
      from reality (e.g. accurate historical/narrative prose that simply
      doesn't read like Reuters wire copy) -- so a limited fact-check
      budget goes furthest there. "supported" segments are lower priority
      for a capped budget, though they are not immune to being wrong
      either (a fluently-written false claim can still read as credible
      style) -- raise max_checks or widen target_assessments to include
      "supported" if that risk matters more to you than the added latency.

    Every result dict's "fact_check" value is either:
      - None: not attempted (skipped by targeting or the cap)
      - a dict shaped like FactCheckDetail (available, verdict, confidence,
        explanation, sources, reason): attempted. `sources` is a list of
        real URLs (Wikipedia + whatever groq/compound-mini's web_search
        step actually returned) -- these are the "webpage links."
    """
    if not fact_checker.available:
        for r in results:
            r["fact_check"] = {
                "available": False,
                "verdict": None,
                "confidence": None,
                "explanation": None,
                "sources": [],
                "reason": fact_checker.reason,
            }
        return results

    checks_used = 0
    for r in results:
        if r["assessment"] not in target_assessments or checks_used >= max_checks:
            r["fact_check"] = None
            continue

        fc = fact_checker.check(r["segment_text"])
        r["fact_check"] = {
            "available": fc.available,
            "verdict": fc.verdict,
            "confidence": fc.confidence,
            "explanation": fc.explanation,
            "sources": fc.sources,
            "reason": fc.reason,
        }
        checks_used += 1

    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate statistics from scored segments."""
    total = len(results)
    if total == 0:
        return {
            "total_segments": 0,
            "avg_confidence": 0.0,
            "supported_pct": 0.0,
            "unsupported_pct": 0.0,
            "uncertain_pct": 0.0,
            "needs_review_pct": 0.0,
        }

    avg_conf = mean(r["confidence_score"] for r in results)
    counts = {"supported": 0, "unsupported": 0, "uncertain": 0, "needs_review": 0}
    for r in results:
        counts[r["assessment"]] += 1

    return {
        "total_segments": total,
        "avg_confidence": round(avg_conf, 4),
        "supported_pct": round(counts["supported"] / total * 100, 1),
        "unsupported_pct": round(counts["unsupported"] / total * 100, 1),
        "uncertain_pct": round(counts["uncertain"] / total * 100, 1),
        "needs_review_pct": round(counts["needs_review"] / total * 100, 1),
    }