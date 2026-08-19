import { useState } from 'react'
import { ChevronDown, ChevronUp, ArrowUpDown, ExternalLink, Search } from 'lucide-react'

const ASSESSMENT_COLORS = {
  supported: { bg: 'bg-[var(--real-dim)]', text: 'text-[var(--real)]', border: 'border-[var(--real)]', label: 'Supported' },
  unsupported: { bg: 'bg-[var(--fake-dim)]', text: 'text-[var(--fake)]', border: 'border-[var(--fake)]', label: 'Unsupported' },
  uncertain: { bg: 'bg-[#3a3020]', text: 'text-[var(--amber)]', border: 'border-[var(--amber)]', label: 'Uncertain' },
  needs_review: { bg: 'bg-[var(--panel-2)]', text: 'text-[var(--text-muted)]', border: 'border-[var(--border-strong)]', label: 'Needs Review' },
}

// Separate from ASSESSMENT_COLORS on purpose -- the fact-check verdict and
// the TF-IDF assessment answer different questions (real-world truth vs.
// stylistic resemblance) and are never visually conflated into one badge.
const FACT_CHECK_COLORS = {
  TRUE: { text: 'text-[var(--real)]', border: 'border-[var(--real)]' },
  FALSE: { text: 'text-[var(--fake)]', border: 'border-[var(--fake)]' },
  MISLEADING: { text: 'text-[var(--amber)]', border: 'border-[var(--amber)]' },
  UNVERIFIED: { text: 'text-[var(--text-muted)]', border: 'border-[var(--border-strong)]' },
}

function FactCheckBlock({ factCheck }) {
  if (!factCheck) return null // not attempted for this segment

  if (!factCheck.available) {
    return (
      <p className="font-mono text-[10px] text-[var(--text-muted)] mt-2 leading-relaxed">
        Fact-check not available: {factCheck.reason || 'unknown reason'}
      </p>
    )
  }

  const colors = FACT_CHECK_COLORS[factCheck.verdict] || FACT_CHECK_COLORS.UNVERIFIED
  const confidencePct = factCheck.confidence != null ? Math.round(factCheck.confidence * 100) : null

  return (
    <div className="mt-2 pt-2 border-t border-[var(--border)]">
      <div className="flex items-center gap-1.5 mb-1">
        <Search className="w-3 h-3 text-[var(--text-muted)]" />
        <span className={`font-mono text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded border ${colors.border} ${colors.text}`}>
          {factCheck.verdict}
        </span>
        {confidencePct != null && (
          <span className="font-mono text-[10px] text-[var(--text-muted)]">{confidencePct}%</span>
        )}
      </div>
      {factCheck.explanation && (
        <p className="text-xs text-[var(--text-primary)] leading-relaxed mb-1">{factCheck.explanation}</p>
      )}
      {factCheck.sources?.length > 0 && (
        <div className="space-y-0.5">
          {factCheck.sources.map((src, i) => (
            <a
              key={i}
              href={src}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 font-mono text-[10px] text-[var(--real)] hover:underline truncate"
            >
              <ExternalLink className="w-2.5 h-2.5 shrink-0" />
              <span className="truncate">{src}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  )
}

function ConfidenceBar({ score }) {
  const pct = Math.round(score * 100)
  let color = 'bg-[var(--real)]'
  if (score <= 0.25) color = 'bg-[var(--fake)]'
  else if (score < 0.75) color = 'bg-[var(--amber)]'

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-[var(--bg)] rounded-full overflow-hidden border border-[var(--border)]">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-xs text-[var(--text-secondary)]">{pct}%</span>
    </div>
  )
}

function SortButton({ column, sortConfig, onSort }) {
  const active = sortConfig.key === column
  return (
    <button
      onClick={() => onSort(column)}
      className="inline-flex items-center gap-1 hover:text-[var(--text-primary)] transition-colors"
    >
      {active ? (
        sortConfig.dir === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
      ) : (
        <ArrowUpDown className="w-3 h-3 opacity-40" />
      )}
    </button>
  )
}

export default function ResultsTable({ segments }) {
  const [sortConfig, setSortConfig] = useState({ key: null, dir: 'asc' })
  const [expandedRows, setExpandedRows] = useState(new Set())

  const toggleRow = (i) => {
    setExpandedRows((prev) => {
      const next = new Set(prev)
      next.has(i) ? next.delete(i) : next.add(i)
      return next
    })
  }

  const handleSort = (key) => {
    setSortConfig((prev) => ({
      key,
      dir: prev.key === key && prev.dir === 'asc' ? 'desc' : 'asc',
    }))
  }

  const sorted = [...segments].sort((a, b) => {
    if (!sortConfig.key) return 0
    const aVal = a[sortConfig.key]
    const bVal = b[sortConfig.key]
    if (typeof aVal === 'number') return sortConfig.dir === 'asc' ? aVal - bVal : bVal - aVal
    return sortConfig.dir === 'asc'
      ? String(aVal).localeCompare(String(bVal))
      : String(bVal).localeCompare(String(aVal))
  })

  return (
    <section className="mb-8">
      <h2 className="font-serif text-xl font-medium mb-3">Claim Results</h2>
      <div className="rounded-xl border border-[var(--border)] overflow-hidden bg-[var(--panel)]">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--border)] bg-[var(--panel-2)]">
                <th className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] text-left px-4 py-3 w-10">
                  #
                </th>
                <th className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] text-left px-4 py-3">
                  Text Segment
                </th>
                <th className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] text-left px-4 py-3 whitespace-nowrap">
                  Assessment <SortButton column="assessment" sortConfig={sortConfig} onSort={handleSort} />
                </th>
                <th className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] text-left px-4 py-3 whitespace-nowrap">
                  Confidence <SortButton column="confidence_score" sortConfig={sortConfig} onSort={handleSort} />
                </th>
                <th className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] text-left px-4 py-3 hidden lg:table-cell">
                  Explanation
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((seg, i) => {
                const colors = ASSESSMENT_COLORS[seg.assessment] || ASSESSMENT_COLORS.needs_review
                const isExpanded = expandedRows.has(i)
                const text = seg.segment_text
                const isLong = text.length > 140
                const displayText = isLong && !isExpanded ? text.slice(0, 140) + '...' : text

                return (
                  <tr
                    key={i}
                    className={`border-b border-[var(--border)] last:border-b-0 ${colors.bg} hover:brightness-110 transition-all`}
                  >
                    <td className="font-mono text-xs text-[var(--text-muted)] px-4 py-3 align-top">
                      {i + 1}
                    </td>
                    <td className="px-4 py-3 align-top max-w-md">
                      <p className="text-[var(--text-primary)] text-sm leading-relaxed">
                        {displayText}
                      </p>
                      {isLong && (
                        <button
                          onClick={() => toggleRow(i)}
                          className="font-mono text-[10px] text-[var(--real)] mt-1 hover:underline"
                        >
                          {isExpanded ? 'Show less' : 'Show full text'}
                        </button>
                      )}
                      <p className="lg:hidden text-xs text-[var(--text-muted)] mt-2 leading-relaxed">
                        {seg.explanation}
                      </p>
                      <div className="lg:hidden">
                        <FactCheckBlock factCheck={seg.fact_check} />
                      </div>
                    </td>
                    <td className="px-4 py-3 align-top">
                      <span className={`inline-block font-mono text-[11px] tracking-wide uppercase px-2.5 py-1 rounded-full border ${colors.border} ${colors.text}`}>
                        {colors.label}
                      </span>
                    </td>
                    <td className="px-4 py-3 align-top">
                      <ConfidenceBar score={seg.confidence_score} />
                    </td>
                    <td className="px-4 py-3 align-top text-xs text-[var(--text-muted)] leading-relaxed hidden lg:table-cell max-w-xs">
                      {seg.explanation}
                      <FactCheckBlock factCheck={seg.fact_check} />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}