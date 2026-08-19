import { Gauge, BarChart3, Layers, Search, ExternalLink, AlertTriangle, CheckCircle2, XCircle, HelpCircle } from 'lucide-react'

const LABEL_STYLES = {
  real: { bg: 'bg-[var(--real-dim)]', text: 'text-[var(--real)]', border: 'border-[var(--real)]', display: 'REAL' },
  fake: { bg: 'bg-[var(--fake-dim)]', text: 'text-[var(--fake)]', border: 'border-[var(--fake)]', display: 'FAKE' },
  uncertain: { bg: 'bg-[var(--amber-dim)]', text: 'text-[var(--amber)]', border: 'border-[var(--amber)]', display: 'UNCERTAIN' },
}

const SOURCE_DISPLAY = {
  weighted_blend: 'Weighted blend',
}

const VERDICT_STYLES = {
  TRUE: { bg: 'bg-[var(--real-dim)]', text: 'text-[var(--real)]', border: 'border-[var(--real)]', icon: CheckCircle2 },
  FALSE: { bg: 'bg-[var(--fake-dim)]', text: 'text-[var(--fake)]', border: 'border-[var(--fake)]', icon: XCircle },
  MISLEADING: { bg: 'bg-[var(--amber-dim)]', text: 'text-[var(--amber)]', border: 'border-[var(--amber)]', icon: AlertTriangle },
  UNVERIFIED: { bg: 'bg-[var(--panel-2)]', text: 'text-[var(--text-muted)]', border: 'border-[var(--border-strong)]', icon: HelpCircle },
}

function Bar({ pct, tone }) {
  const fill = {
    real: 'bg-[var(--real)]', fake: 'bg-[var(--fake)]', uncertain: 'bg-[var(--amber)]', neutral: 'bg-[var(--text-muted)]',
  }[tone] || 'bg-[var(--text-muted)]'
  return (
    <div className="h-2 bg-[var(--bg)] rounded-full overflow-hidden border border-[var(--border)] flex-1">
      <div className={`h-full ${fill} rounded-full transition-all duration-600`} style={{ width: `${Math.round(pct * 100)}%` }} />
    </div>
  )
}

/**
 * Full 3-signal ensemble result: one final verdict up top (with the
 * "which signal decided this" explanation always visible), then each of
 * the three individual model outputs shown separately underneath so the
 * user can see exactly which model predicted what, then the web
 * fact-check evidence -- shown whenever it ran, with sources always
 * surfaced regardless of whether the claim was supported or not.
 */
export default function CombinedPanel({ result }) {
  if (!result) return null

  const finalStyles = LABEL_STYLES[result.final_label] || LABEL_STYLES.uncertain
  const finalPct = Math.round(result.final_confidence * 100)
  const weights = result.weights_used || {}
  const pct = (w) => (w != null ? `${Math.round(w * 100)}%` : null)

  const tfidf = result.tfidf
  const tfidfTone = tfidf.label === 'real' ? 'real' : 'fake'

  const liar = result.liar
  const liarTone = liar ? (liar.direction === 'real' ? 'real' : liar.direction === 'fake' ? 'fake' : 'uncertain') : 'neutral'

  const fc = result.fact_check
  const fcStyles = fc?.verdict ? (VERDICT_STYLES[fc.verdict] || VERDICT_STYLES.UNVERIFIED) : null
  const FcIcon = fcStyles?.icon

  return (
    <div className="mt-5 space-y-5">
      {/* ---- FINAL COMBINED VERDICT ---- */}
      <div className="rounded-xl border border-[var(--border-strong)] bg-[var(--panel-2)] p-5">
        <div className="flex items-center gap-2 mb-3">
          <Gauge className="w-4 h-4 text-[var(--text-secondary)]" />
          <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-secondary)]">
            Final ensemble verdict
          </span>
        </div>

        <div className="flex items-center gap-3 mb-3 flex-wrap">
          <span className={`font-mono text-base font-medium px-3.5 py-1.5 rounded ${finalStyles.bg} ${finalStyles.text} border ${finalStyles.border}`}>
            {finalStyles.display}
          </span>
          <span className="font-mono text-sm text-[var(--text-secondary)]">
            confidence {finalPct}%
          </span>
          <span className="font-mono text-[10px] text-[var(--text-muted)] uppercase tracking-wide border border-[var(--border)] rounded px-2 py-0.5">
            {SOURCE_DISPLAY[result.final_source] || result.final_source}
          </span>
        </div>

        {/* Weight legend -- exactly how much each available signal counted */}
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {weights.web != null && (
            <span className="font-mono text-[10px] px-2 py-0.5 rounded border border-[var(--border)] text-[var(--text-secondary)]">
              🔍 Web {pct(weights.web)}
            </span>
          )}
          <span className="font-mono text-[10px] px-2 py-0.5 rounded border border-[var(--border)] text-[var(--text-secondary)]">
            📊 TF-IDF {pct(weights.tfidf)}
          </span>
          {weights.liar != null && (
            <span className="font-mono text-[10px] px-2 py-0.5 rounded border border-[var(--border)] text-[var(--text-secondary)]">
              🧠 LIAR {pct(weights.liar)}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 mb-3">
          <Bar pct={result.final_confidence} tone={result.final_label} />
          <span className="font-mono text-xs text-[var(--text-secondary)] w-10 text-right">{finalPct}%</span>
        </div>

        <p className="text-sm text-[var(--text-primary)] leading-relaxed">
          {result.explanation}
        </p>
      </div>

      {/* ---- INDIVIDUAL SIGNALS, so the user can see who said what ---- */}
      <div>
        <p className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)] mb-2">
          Individual model breakdown
        </p>
        <div className="grid gap-3 md:grid-cols-3">
          {/* TF-IDF */}
          <div className="rounded-lg border border-[var(--border)] bg-[var(--panel)] p-4">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">
                TF-IDF classifier
              </span>
              <span className="font-mono text-[9px] text-[var(--text-muted)] ml-auto border border-[var(--border)] rounded px-1.5 py-0.5">
                weight {pct(weights.tfidf)}
              </span>
            </div>
            <div className="flex items-center gap-2 mb-2">
              <span className={`font-mono text-xs font-medium px-2 py-0.5 rounded ${LABEL_STYLES[tfidfTone].bg} ${LABEL_STYLES[tfidfTone].text} border ${LABEL_STYLES[tfidfTone].border}`}>
                {tfidf.label.toUpperCase()}
              </span>
              <span className="font-mono text-xs text-[var(--text-secondary)]">{Math.round(tfidf.confidence * 100)}%</span>
            </div>
            <Bar pct={tfidf.confidence} tone={tfidfTone} />
            <p className="font-mono text-[10px] text-[var(--text-muted)] mt-2">{tfidf.model_used}</p>
          </div>

          {/* LIAR / RoBERTa */}
          <div className="rounded-lg border border-[var(--border)] bg-[var(--panel)] p-4">
            <div className="flex items-center gap-2 mb-2">
              <Layers className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">
                RoBERTa / LIAR
              </span>
              {weights.liar != null && (
                <span className="font-mono text-[9px] text-[var(--text-muted)] ml-auto border border-[var(--border)] rounded px-1.5 py-0.5">
                  weight {pct(weights.liar)}
                </span>
              )}
            </div>
            {liar ? (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <span className={`font-mono text-xs font-medium px-2 py-0.5 rounded ${LABEL_STYLES[liarTone].bg} ${LABEL_STYLES[liarTone].text} border ${LABEL_STYLES[liarTone].border}`}>
                    {liar.bucket.replace('-', ' ').toUpperCase()}
                  </span>
                  <span className="font-mono text-xs text-[var(--text-secondary)]">{Math.round(liar.confidence * 100)}%</span>
                </div>
                <Bar pct={liar.confidence} tone={liarTone} />
                <p className="font-mono text-[10px] text-[var(--text-muted)] mt-2">
                  6-class: {liar.label} &middot; {liar.model_version}
                </p>
              </>
            ) : (
              <p className="font-mono text-[10px] text-[var(--text-muted)] leading-relaxed">
                Not used ({result.liar_gate_status || 'gate closed'})
              </p>
            )}
          </div>

          {/* Web fact-check */}
          <div className="rounded-lg border border-[var(--border)] bg-[var(--panel)] p-4">
            <div className="flex items-center gap-2 mb-2">
              <Search className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-secondary)]">
                Web fact-check
              </span>
              {weights.web != null && (
                <span className="font-mono text-[9px] text-[var(--text-muted)] ml-auto border border-[var(--border)] rounded px-1.5 py-0.5">
                  weight {pct(weights.web)}
                </span>
              )}
            </div>
            {fc?.available && fc.confidence != null ? (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <span className={`font-mono text-xs font-medium px-2 py-0.5 rounded ${fcStyles.bg} ${fcStyles.text} border ${fcStyles.border}`}>
                    {fc.verdict}
                  </span>
                  <span className="font-mono text-xs text-[var(--text-secondary)]">{Math.round(fc.confidence * 100)}%</span>
                </div>
                <Bar pct={fc.confidence} tone={fc.verdict === 'TRUE' ? 'real' : fc.verdict === 'FALSE' ? 'fake' : 'uncertain'} />
              </>
            ) : (
              <p className="font-mono text-[10px] text-[var(--text-muted)] leading-relaxed">
                {result.web_search_triggered
                  ? (fc?.reason || 'Web check ran but returned no usable verdict.')
                  : 'Not run this time.'}
              </p>
            )}
          </div>
        </div>

        {/* Why web search did/didn't run -- always visible, not tucked away */}
        <p className="font-mono text-[10px] text-[var(--text-muted)] mt-2 leading-relaxed">
          {result.web_search_triggered ? '🔍 ' : '⏭ '}{result.web_search_trigger_reason}
        </p>
      </div>

      {/* ---- WEB RESULTS: shown whenever the fact-check ran, with sources
             always surfaced -- and extra emphasis (confidence + explanation
             front and center) when the claim was NOT supported. ---- */}
      {fc?.available && (
        <div
          className={`rounded-lg border p-4 ${
            fc.verdict === 'FALSE'
              ? 'border-[var(--fake)] bg-[var(--fake-dim)]'
              : fc.verdict === 'MISLEADING'
              ? 'border-[var(--amber)] bg-[var(--amber-dim)]'
              : 'border-[var(--border-strong)] bg-[var(--panel-2)]'
          }`}
        >
          <div className="flex items-center gap-2 mb-3">
            {FcIcon && <FcIcon className={`w-4 h-4 ${fcStyles.text}`} />}
            <span className={`font-mono text-[11px] uppercase tracking-wider ${fcStyles?.text || 'text-[var(--text-secondary)]'}`}>
              {fc.verdict === 'FALSE' || fc.verdict === 'MISLEADING'
                ? 'Not supported by web evidence'
                : fc.verdict === 'TRUE'
                ? 'Supported by web evidence'
                : 'Web evidence'}
            </span>
            {fc.confidence != null && (
              <span className="font-mono text-[11px] text-[var(--text-secondary)] ml-auto">
                {Math.round(fc.confidence * 100)}% confidence
              </span>
            )}
          </div>

          {fc.explanation && (
            <p className="text-sm text-[var(--text-primary)] leading-relaxed mb-3">
              {fc.explanation}
            </p>
          )}

          {fc.sources?.length > 0 ? (
            <div className="space-y-1">
              <p className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">
                Sources checked
              </p>
              {fc.sources.map((src, i) => (
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
          ) : (
            <p className="font-mono text-[10px] text-[var(--text-muted)]">No source URLs were returned.</p>
          )}
        </div>
      )}

      {fc && !fc.available && (
        <div className="rounded-lg border border-[var(--border-strong)] bg-[var(--panel-2)] p-4">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle className="w-3.5 h-3.5 text-[var(--text-muted)]" />
            <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
              Web fact-check unavailable
            </span>
          </div>
          <p className="font-mono text-xs text-[var(--text-muted)] leading-relaxed">
            {fc.reason || 'The web-search fact-check could not run.'}
          </p>
        </div>
      )}
    </div>
  )
}
