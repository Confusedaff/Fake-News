import { Layers, AlertTriangle } from 'lucide-react'

const BUCKET_STYLES = {
  'leans-real': { bg: 'bg-[var(--real-dim)]', text: 'text-[var(--real)]', border: 'border-[var(--real)]', display: 'Leans Real' },
  'leans-fake': { bg: 'bg-[var(--fake-dim)]', text: 'text-[var(--fake)]', border: 'border-[var(--fake)]', display: 'Leans Fake' },
  'uncertain': { bg: 'bg-[var(--amber-dim)]', text: 'text-[var(--amber)]', border: 'border-[var(--amber)]', display: 'Uncertain' },
}

// Renders the secondary RoBERTa/LIAR signal. Kept visually and structurally
// separate from the primary TF-IDF label above it -- the LIAR bucket is a
// gated, weaker-confidence secondary opinion (eval_f1_macro=0.445 on 6
// classes), never blended into the main label/confidence, matching the
// backend's own never-override design.
export default function EnsemblePanel({ liarSignalUsed, liarDetail, liarGateStatus, unverifiable }) {
  if (!liarSignalUsed || !liarDetail) {
    return (
      <div className="mt-4 rounded-lg border border-[var(--border-strong)] bg-[var(--panel-2)] p-4">
        <div className="flex items-center gap-2 mb-1">
          <AlertTriangle className="w-3.5 h-3.5 text-[var(--text-muted)]" />
          <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
            LIAR ensemble signal unavailable
          </span>
        </div>
        <p className="font-mono text-xs text-[var(--text-muted)] leading-relaxed">
          {liarGateStatus || 'The secondary RoBERTa/LIAR model did not produce a signal for this request.'}
        </p>
      </div>
    )
  }

  const styles = BUCKET_STYLES[liarDetail.bucket] || BUCKET_STYLES.uncertain
  const confidencePct = liarDetail.confidence != null ? Math.round(liarDetail.confidence * 100) : null

  return (
    <div className="mt-4 rounded-lg border border-[var(--border-strong)] bg-[var(--panel-2)] p-4">
      <div className="flex items-center gap-2 mb-3">
        <Layers className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
        <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-secondary)]">
          LIAR ensemble &middot; secondary style signal, gated
        </span>
      </div>

      <div className="flex items-center gap-3 mb-2">
        <span
          className={`font-mono text-sm font-medium px-3 py-1 rounded ${styles.bg} ${styles.text} border ${styles.border}`}
        >
          {styles.display}
        </span>
        {confidencePct != null && (
          <span className="font-mono text-sm text-[var(--text-secondary)]">
            confidence {confidencePct}%
          </span>
        )}
        <span className="font-mono text-[10px] text-[var(--text-muted)] uppercase tracking-wide">
          {liarDetail.model_version}
        </span>
      </div>

      <p className="font-mono text-[11px] text-[var(--text-muted)] leading-relaxed">
        6-class label: <span className="text-[var(--text-secondary)]">{liarDetail.label}</span>
        {unverifiable && (
          <span className="text-[var(--amber)]"> &middot; flagged unverifiable (lands in the uncertain middle buckets)</span>
        )}
      </p>

      <p className="font-mono text-[10px] text-[var(--text-muted)] mt-2 leading-relaxed">
        This is a weaker, secondary opinion (trained on LIAR, gated by its own validation
        score) -- never blended with the primary classifier's label above.
      </p>
    </div>
  )
}
