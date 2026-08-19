import { Search, AlertTriangle, ExternalLink } from 'lucide-react'

const VERDICT_STYLES = {
  TRUE: { bg: 'bg-[var(--real-dim)]', text: 'text-[var(--real)]', border: 'border-[var(--real)]' },
  FALSE: { bg: 'bg-[var(--fake-dim)]', text: 'text-[var(--fake)]', border: 'border-[var(--fake)]' },
  MISLEADING: { bg: 'bg-[var(--amber-dim)]', text: 'text-[var(--amber)]', border: 'border-[var(--amber)]' },
  UNVERIFIED: { bg: 'bg-[var(--panel-2)]', text: 'text-[var(--text-muted)]', border: 'border-[var(--border-strong)]' },
}

// Renders the agentic web-search fact-check result. Deliberately separate
// from the TF-IDF label/confidence block above it -- these two signals
// answer different questions (style vs. real-world truth) and are never
// blended into one number, matching the backend's own design philosophy.
export default function FactCheckPanel({ factCheck }) {
  if (!factCheck) return null

  if (!factCheck.available) {
    return (
      <div className="mt-4 rounded-lg border border-[var(--border-strong)] bg-[var(--panel-2)] p-4">
        <div className="flex items-center gap-2 mb-1">
          <AlertTriangle className="w-3.5 h-3.5 text-[var(--text-muted)]" />
          <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
            Fact-check unavailable
          </span>
        </div>
        <p className="font-mono text-xs text-[var(--text-muted)] leading-relaxed">
          {factCheck.reason || 'The web-search fact-check could not run.'}
        </p>
      </div>
    )
  }

  const styles = VERDICT_STYLES[factCheck.verdict] || VERDICT_STYLES.UNVERIFIED
  const confidencePct = factCheck.confidence != null ? Math.round(factCheck.confidence * 100) : null

  return (
    <div className="mt-4 rounded-lg border border-[var(--border-strong)] bg-[var(--panel-2)] p-4">
      <div className="flex items-center gap-2 mb-3">
        <Search className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
        <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-secondary)]">
          Real-world fact-check &middot; live web search
        </span>
      </div>

      <div className="flex items-center gap-3 mb-3">
        <span
          className={`font-mono text-sm font-medium px-3 py-1 rounded ${styles.bg} ${styles.text} border ${styles.border}`}
        >
          {factCheck.verdict}
        </span>
        {confidencePct != null && (
          <span className="font-mono text-sm text-[var(--text-secondary)]">
            confidence {confidencePct}%
          </span>
        )}
      </div>

      {factCheck.explanation && (
        <p className="text-sm text-[var(--text-primary)] leading-relaxed mb-3">
          {factCheck.explanation}
        </p>
      )}

      {factCheck.sources?.length > 0 && (
        <div className="space-y-1">
          <p className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">
            Sources
          </p>
          {factCheck.sources.map((src, i) => (
            <a
              key={i}
              href={src}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 font-mono text-[11px] text-[var(--real)] hover:underline truncate"
            >
              <ExternalLink className="w-3 h-3 shrink-0" />
              <span className="truncate">{src}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  )
}
