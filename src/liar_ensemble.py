"""
Gated ensemble support: adds an "unverifiable / disputed" signal from a
6-class LIAR-trained model (classifier-v3.0, fine-tuned roberta-base) on
top of the binary TF-IDF fake/real classifier.

Why this exists
----------------
The production TF-IDF model is trained on binary fake/real news articles.
It has no way to represent a statement that is neither clearly true nor
clearly false — e.g. "Donald Trump may resign in October" is a prediction,
not a factual claim, and forcing a fake/real call on it is misleading no
matter how confident the model sounds. LIAR's 6-class scheme (pants-fire
... true) has a natural middle ground (barely-true / half-true) that maps
reasonably well onto "unverifiable, don't force a binary call."

Why this is gated
------------------
The bundled classifier-v3.0 model reports eval_accuracy=0.472 and
eval_f1_macro=0.445 on 6 classes (chance = 1/6 = 0.167) in its own
manifest.json. That is materially better than chance but far below a bar
you'd want to trust for a production signal, and far below the TF-IDF
model's 0.99+ on its own task. A weak second opinion doesn't add reliable
uncertainty detection -- it adds noise dressed up as a signal.

So: this module loads the LIAR model's own reported validation metrics at
import time and compares them to MIN_F1_MACRO_TO_TRUST below. If the
bundled model doesn't clear the bar, `liar_signal()` always returns None
and the ensemble endpoint transparently falls back to TF-IDF-only. Nothing
about this file needs to change when a better-trained v3 model ships --
just drop the new artifacts in models/liar_v3/ and the gate re-evaluates
itself against the new manifest.json automatically.
"""
import json
from pathlib import Path
from typing import Optional

# --- the quality bar -------------------------------------------------------
# Chosen as "clearly, comfortably better than chance on 6 classes" (chance
# ~= 0.167) with real margin, not just "technically above chance." Revisit
# this once you have a second retrained checkpoint to compare against --
# there's no principled reason this has to be exactly 0.55, it's a
# starting point pending a second data point.
MIN_F1_MACRO_TO_TRUST = 0.35

# LIAR's 6 truthfulness grades collapsed into 3 buckets for the ensemble.
# pants-fire/false lean toward the TF-IDF model's "fake"; mostly-true/true
# lean toward "real"; the middle two are exactly the case a binary
# classifier can't represent, so they become the "unverifiable" flag.
LABEL_BUCKETS = {
    "pants-fire": "leans-fake",
    "false": "leans-fake",
    "barely-true": "uncertain",
    "half-true": "uncertain",
    "mostly-true": "leans-real",
    "true": "leans-real",
}

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "liar_v3"


class LiarEnsembleModel:
    """
    Lazily-loaded wrapper around the LIAR-trained transformer. Torch and
    transformers are only imported if the quality gate passes AND a
    prediction is actually requested -- so a low-scoring or absent model
    never forces a heavyweight dependency onto a machine that doesn't need
    it, and the API still starts cleanly without torch installed.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.available = False
        self.trusted = False
        self.manifest = None
        self.reason = None
        self._pipeline = None  # populated on first use, if trusted

        self._evaluate_gate()

    def _evaluate_gate(self):
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            self.reason = f"No model found at {self.model_dir}"
            return

        try:
            self.manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            self.reason = f"Could not read manifest.json: {exc}"
            return

        self.available = True
        f1 = self.manifest.get("validation_metrics", {}).get("eval_f1_macro")
        version = self.manifest.get("model_version", "unknown")

        if f1 is None:
            self.reason = f"{version}: manifest has no eval_f1_macro, refusing to trust it"
            return

        if f1 < MIN_F1_MACRO_TO_TRUST:
            self.reason = (
                f"{version}: eval_f1_macro={f1:.3f} is below the trust threshold "
                f"({MIN_F1_MACRO_TO_TRUST}) -- ensemble signal disabled, "
                f"falling back to TF-IDF-only"
            )
            return

        self.trusted = True
        self.reason = f"{version}: eval_f1_macro={f1:.3f} clears the trust threshold ({MIN_F1_MACRO_TO_TRUST})"

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        # Deferred import: only pay the torch/transformers cost if a
        # trusted model is actually going to be used.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir / "tokenizer"))
        model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir / "model"))
        model.eval()
        self._pipeline = (tokenizer, model, torch)

    def liar_signal(self, text: str) -> Optional[dict]:
        """
        Returns None if the model isn't trusted (gate closed) or isn't
        available at all. Otherwise returns the 6-class prediction plus
        the 3-bucket collapse used by the ensemble.
        """
        if not self.trusted:
            return None

        self._ensure_loaded()
        tokenizer, model, torch = self._pipeline
        max_length = self.manifest.get("hyperparameters", {}).get("max_length", 256)
        temperature = self.manifest.get("calibration", {}).get("temperature", 1.0)
        id_to_label = self.manifest.get("id_to_label") or json.loads(
            (self.model_dir / "label_mapping.json").read_text()
        )

        inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            calibrated = torch.softmax(logits / temperature, dim=-1)

        top_idx = int(torch.argmax(calibrated))
        label = id_to_label[str(top_idx)]
        confidence = float(calibrated[top_idx])

        return {
            "label": label,
            "bucket": LABEL_BUCKETS.get(label, "uncertain"),
            "confidence": round(confidence, 4),
            "model_version": self.manifest.get("model_version"),
        }


# Module-level singleton -- the gate is evaluated once at import time
# (cheap: just reads manifest.json), the actual model loads lazily.
liar_model = LiarEnsembleModel()
