const COLORS = {
  supported: { bg: 'bg-[var(--real)]', label: 'Supported' },
  unsupported: { bg: 'bg-[var(--fake)]', label: 'Unsupported' },
  uncertain: { bg: 'bg-[var(--amber)]', label: 'Uncertain' },
  needs_review: { bg: 'bg-[var(--text-muted)]', label: 'Needs Review' },
}

export default function SummaryStats({ summary }) {
  if (!summary) return null

  const bars = [
    { key: 'supported', pct: summary.supported_pct },
    { key: 'unsupported', pct: summary.unsupported_pct },
    { key: 'uncertain', pct: summary.uncertain_pct },
    { key: 'needs_review', pct: summary.needs_review_pct },
  ]

  return (
    <section className="mb-8">
      <h2 className="font-serif text-xl font-medium mb-3">Summary</h2>
      <div className="rounded-xl border border-[var(--border)] bg-[var(--panel)] p-5">
        <div className="flex flex-wrap gap-6 mb-5">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">
              Total Segments
            </p>
            <p className="font-mono text-2xl text-[var(--text-primary)]">{summary.total_segments}</p>
          </div>
          <div>
            <p className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">
              Avg Confidence
            </p>
            <p className="font-mono text-2xl text-[var(--text-primary)]">
              {Math.round(summary.avg_confidence * 100)}%
            </p>
          </div>
        </div>

        {/* Stacked bar */}
        <div className="mb-3">
          <div className="h-3 bg-[var(--bg)] rounded-full overflow-hidden border border-[var(--border)] flex">
            {bars.map((b) =>
              b.pct > 0 ? (
                <div
                  key={b.key}
                  className={`${COLORS[b.key].bg} h-full transition-all duration-500`}
                  style={{ width: `${b.pct}%` }}
                  title={`${COLORS[b.key].label}: ${b.pct}%`}
                />
              ) : null
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4">
          {bars.map((b) => (
            <div key={b.key} className="flex items-center gap-2">
              <div className={`w-2.5 h-2.5 rounded-full ${COLORS[b.key].bg}`} />
              <span className="font-mono text-xs text-[var(--text-secondary)]">
                {COLORS[b.key].label}: <strong className="text-[var(--text-primary)]">{b.pct}%</strong>
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
