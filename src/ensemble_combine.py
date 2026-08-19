"""
Full 3-signal ensemble: TF-IDF classifier + RoBERTa/LIAR style signal +
agentic web-search fact-check, combined into ONE final verdict + confidence,
with every individual signal preserved alongside it for transparency.

Combination rule: weighted blend, web check weighted highest
--------------------------------------------------------------
The three signals answer genuinely different questions:
  - TF-IDF:      "does this text's STYLE resemble known credible/non-credible
                  sources?" -- fast, local, 0.99+ validated accuracy on its
                  own narrow task, but purely stylistic.
  - LIAR/RoBERTa: a second, weaker stylistic opinion with a natural
                  "unverifiable / half-true" middle ground the binary
                  TF-IDF model can't express (see src/liar_ensemble.py).
                  Only consulted at all if its own validation F1 clears
                  MIN_F1_MACRO_TO_TRUST.
  - Web fact-check: "is this claim actually TRUE, right now, per live
                  search evidence?" -- the only one of the three that
                  looks at the real world instead of writing style.

Each signal is converted onto a common 0..1 "probability the claim is
real" scale, then combined as a weighted average using WEIGHTS below.
Web fact-check gets the highest weight (it's grounded in real-world
evidence, not just writing style) but TF-IDF and LIAR still contribute
-- a strong live source can be outvoted by two style classifiers that
both disagree with it, rather than being silently ignored, and a
MISLEADING web result still pulls the blend toward neutral (0.5)
rather than dropping out.

If a signal is unavailable (LIAR gate closed, web check failed to run,
errored, or came back UNVERIFIED), its weight is dropped and the
remaining weights are renormalized to sum to 1, so the two live
signals still combine sensibly on their own.

The final probability maps back to a label: "real" above the upper
band, "fake" below the lower band, "uncertain" in between (the blend
didn't lean either way with enough margin to call it).
"""
from dataclasses import dataclass, field
from typing import Optional

# Weight given to each signal in the blend when all three are available.
# Web fact-check dominates (it's the only real-world-grounded signal) but
# TF-IDF and LIAR still meaningfully move the final number.
WEIGHTS = {
    "web": 0.60,
    "tfidf": 0.25,
    "liar": 0.15,
}

# How far the blended probability has to sit from 0.5 before we call it
# "real"/"fake" outright rather than "uncertain". A tight band would call
# every near-coin-flip blend a confident verdict; a wide band would report
# "uncertain" too often. 0.05 means anything within 45-55% stays uncertain.
UNCERTAIN_BAND = 0.05

CONFIDENCE_TRIGGER = 0.90  # kept for reference/tests; no longer gates the web search call itself


@dataclass
class CombinedResult:
    # --- final, combined call -----------------------------------------
    final_label: str                 # "real" | "fake" | "uncertain"
    final_confidence: float
    final_source: str                # "weighted_blend" -- see weights_used for the actual mix
    explanation: str                 # human-readable reason for the final call
    weights_used: dict = field(default_factory=dict)  # which signals contributed & at what weight

    # --- individual signals, always returned in full ------------------
    tfidf: dict = field(default_factory=dict)
    liar: Optional[dict] = None      # None if the LIAR gate is closed
    fact_check: Optional[dict] = None  # None if the web check was never triggered

    # --- transparency into the process itself --------------------------
    web_search_triggered: bool = False
    web_search_trigger_reason: str = ""


def _liar_direction(liar_detail: Optional[dict]) -> Optional[str]:
    """Collapse a LIAR bucket into 'real' / 'fake' / 'uncertain' so it can
    be compared against the TF-IDF label directly."""
    if not liar_detail:
        return None
    bucket = liar_detail.get("bucket")
    if bucket == "leans-real":
        return "real"
    if bucket == "leans-fake":
        return "fake"
    return "uncertain"


def _liar_real_probability(liar_detail: Optional[dict]) -> Optional[float]:
    """Map the LIAR signal onto the same 0..1 'P(real)' scale as TF-IDF
    and the web check, so all three can be weight-averaged directly."""
    if not liar_detail:
        return None
    bucket = liar_detail.get("bucket")
    conf = liar_detail.get("confidence", 0.5)
    if bucket == "leans-real":
        return 0.5 + conf / 2   # confident leans-real -> close to 1.0
    if bucket == "leans-fake":
        return 0.5 - conf / 2   # confident leans-fake -> close to 0.0
    return 0.5                  # uncertain/half-true -> neutral


def _web_real_probability(fact_check_result) -> Optional[float]:
    """Map the web fact-check verdict onto the same 0..1 'P(real)' scale.
    Only usable, available verdicts contribute -- UNVERIFIED or an
    unavailable check contributes nothing (its weight gets redistributed
    to the other signals rather than forced to 0.5, since 0.5 would
    silently drag every blend toward uncertain on every unavailable
    fact-check, which is a real-world availability constraint in this
    repo -- see fact_check.py -- not a genuine 'no evidence either way'
    signal)."""
    if fact_check_result is None or not fact_check_result.available or fact_check_result.confidence is None:
        return None
    verdict = fact_check_result.verdict
    conf = fact_check_result.confidence
    if verdict == "TRUE":
        return 0.5 + conf / 2
    if verdict == "FALSE":
        return 0.5 - conf / 2
    if verdict == "MISLEADING":
        return 0.5  # technically-true-but-deceptive pulls toward neutral, doesn't drop out
    return None  # UNVERIFIED -- no usable read, drop from the blend


def decide_web_search_trigger(tfidf_label: str, tfidf_confidence: float, liar_detail: Optional[dict]) -> tuple[bool, str]:
    """Returns (should_run, human_readable_reason).

    Web search always runs, independent of what TF-IDF/LIAR say. Kept as
    a function (rather than inlined) so the "why" is still reported back
    to the caller/frontend, and so the trigger policy stays a single,
    easily-changed place if that ever needs to be revisited.
    """
    return True, (
        "Web search runs on every request to check the claim against live, "
        "real-world evidence, regardless of the TF-IDF/LIAR classifier results."
    )


def combine(
    tfidf_label: str,
    tfidf_confidence: float,
    fake_probability: float,
    real_probability: float,
    model_used: str,
    liar_detail: Optional[dict],
    liar_gate_status: str,
    fact_check_result,  # a fact_check.FactCheckResult, or None if never run
    web_search_triggered: bool,
    web_search_trigger_reason: str,
) -> CombinedResult:
    tfidf_block = {
        "label": tfidf_label,
        "confidence": round(tfidf_confidence, 4),
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(real_probability, 4),
        "model_used": model_used,
    }

    liar_block = None
    if liar_detail is not None:
        liar_block = {**liar_detail, "direction": _liar_direction(liar_detail)}

    fc_block = None
    if fact_check_result is not None:
        fc_block = {
            "available": fact_check_result.available,
            "verdict": fact_check_result.verdict,
            "confidence": fact_check_result.confidence,
            "explanation": fact_check_result.explanation,
            "sources": fact_check_result.sources,
            "reason": fact_check_result.reason,
        }

    # --- build the weighted blend ---------------------------------------
    # Each signal contributes a "P(real)" in 0..1 plus its configured
    # weight; unavailable signals are simply omitted, and the remaining
    # weights are renormalized so they still sum to 1.
    contributions = {"tfidf": real_probability}  # TF-IDF's own softmax P(real) -- always available

    liar_p = _liar_real_probability(liar_detail)
    if liar_p is not None:
        contributions["liar"] = liar_p

    web_p = _web_real_probability(fact_check_result)
    if web_p is not None:
        contributions["web"] = web_p

    raw_weight_total = sum(WEIGHTS[k] for k in contributions)
    weights_used = {k: round(WEIGHTS[k] / raw_weight_total, 4) for k in contributions}

    blended_p_real = sum(contributions[k] * weights_used[k] for k in contributions)

    # --- map the blended probability to a label + confidence ------------
    if blended_p_real >= 0.5 + UNCERTAIN_BAND:
        final_label = "real"
        final_confidence = round(blended_p_real, 4)
    elif blended_p_real <= 0.5 - UNCERTAIN_BAND:
        final_label = "fake"
        final_confidence = round(1 - blended_p_real, 4)
    else:
        final_label = "uncertain"
        # Report how far from a genuine 50/50 split the blend actually
        # landed, so "uncertain" still carries a meaningful number rather
        # than always showing 50%.
        final_confidence = round(0.5 + abs(blended_p_real - 0.5), 4)

    # --- build a readable explanation of exactly what went into it ------
    parts = [f"TF-IDF {weights_used['tfidf']:.0%} (P(real)={real_probability:.0%})"]
    if "liar" in contributions:
        parts.append(f"LIAR {weights_used['liar']:.0%} (P(real)={liar_p:.0%})")
    else:
        parts.append("LIAR not included (gate closed)")
    if "web" in contributions:
        verdict_note = fact_check_result.verdict
        parts.append(f"web fact-check {weights_used['web']:.0%} (verdict={verdict_note}, P(real)={web_p:.0%})")
    else:
        reason = "UNVERIFIED" if (fact_check_result is not None and fact_check_result.available) else (
            fact_check_result.reason if fact_check_result is not None else "did not run"
        )
        parts.append(f"web fact-check not included ({reason})")

    explanation = (
        f"Weighted blend of available signals, web fact-check weighted highest: "
        f"{'; '.join(parts)}. Combined P(real) = {blended_p_real:.0%}."
    )

    return CombinedResult(
        final_label=final_label,
        final_confidence=final_confidence,
        final_source="weighted_blend",
        explanation=explanation,
        weights_used=weights_used,
        tfidf=tfidf_block, liar=liar_block, fact_check=fc_block,
        web_search_triggered=web_search_triggered,
        web_search_trigger_reason=web_search_trigger_reason,
    )
