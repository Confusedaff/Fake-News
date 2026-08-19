import { useState } from 'react'
import { Send, RotateCcw, Search, Layers, Gauge } from 'lucide-react'
import api from '../api'
import FactCheckPanel from './FactCheckPanel'
import EnsemblePanel from './EnsemblePanel'
import CombinedPanel from './CombinedPanel'

const MODES = [
  { id: 'classifier', label: 'Classifier only', endpoint: '/predict' },
  { id: 'ensemble', label: '+ LIAR ensemble', endpoint: '/predict/ensemble' },
  { id: 'factcheck', label: '+ Web fact-check', endpoint: '/predict/factcheck' },
  { id: 'combined', label: 'Full ensemble', endpoint: '/predict/combined' },
]

const EMPTY_RESULTS_BY_MODE = { classifier: null, ensemble: null, factcheck: null, combined: null }

// Badge color tokens, matching the conventions already used in
// EnsemblePanel (bucket) and FactCheckPanel (verdict) so the top badge reads
// consistently with the detail panels beneath it.
const TONE_STYLES = {
  real: 'bg-[var(--real-dim)] text-[var(--real)] border border-[var(--real)]',
  fake: 'bg-[var(--fake-dim)] text-[var(--fake)] border border-[var(--fake)]',
  amber: 'bg-[var(--amber-dim)] text-[var(--amber)] border border-[var(--amber)]',
  neutral: 'bg-[var(--panel-2)] text-[var(--text-muted)] border border-[var(--border-strong)]',
}
// Solid-fill versions for the single confidence bar (non-classifier modes).
const TONE_BAR_FILL = {
  real: 'bg-[var(--real)]',
  fake: 'bg-[var(--fake)]',
  amber: 'bg-[var(--amber)]',
  neutral: 'bg-[var(--text-muted)]',
}

/**
 * Turns a raw API response for a given mode into what the top of the result
 * block should show.
 *
 * Only the classifier's fake/real numbers are a genuine complementary pair
 * (softmax over 2 classes, they sum to 1) -- that's the only case where a
 * FAKE-vs-REAL two-bar split means anything. The other two modes each
 * report a *single* confidence in a *single* predicted class out of several
 * (LIAR: 6-class; fact-check: TRUE/FALSE/MISLEADING/UNVERIFIED), so
 * "1 - confidence" there is NOT "probability of the opposite side" -- it's
 * just "probability mass on all other classes combined," several of which
 * can point the same direction as the top class. Treating it as a
 * complementary pair previously produced results like a LIAR bucket of
 * leans-real at 29% confidence rendering as FAKE 71% / REAL 29% --
 * contradicting its own badge. So: classifier mode (and any fallback to
 * the classifier) gets the real two-bar split; ensemble/factcheck's own
 * signal gets a single confidence bar in the direction's color instead of a
 * fabricated opposite-side number.
 */
function deriveDisplay(mode, raw) {
  if (!raw) return null

  const classifierFallback = (note) => ({
    hasBinarySplit: true,
    fakePct: raw.fake_probability,
    realPct: raw.real_probability,
    badgeLabel: raw.label.toUpperCase(),
    tone: raw.label === 'real' ? 'real' : 'fake',
    confidence: Math.max(raw.fake_probability, raw.real_probability),
    modelUsed: raw.model_used,
    note,
  })

  const singleSignal = (badgeLabel, tone, confidence, modelUsed, note) => ({
    hasBinarySplit: false,
    signalPct: confidence,
    badgeLabel, tone, confidence, modelUsed, note,
  })

  if (mode === 'classifier') {
    return classifierFallback(null)
  }

  if (mode === 'ensemble') {
    if (!raw.liar_signal_used || !raw.liar_detail) {
      return classifierFallback(
        `LIAR ensemble signal unavailable (${raw.liar_gate_status || 'gate closed'}) -- showing classifier-only score`
      )
    }
    const { bucket, confidence, model_version } = raw.liar_detail
    const modelUsed = model_version || 'RoBERTa/LIAR'
    const baseNote = 'LIAR is a 6-class model -- this is its confidence in its single top class, not a fake/real split.'
    if (bucket === 'leans-real') return singleSignal('LEANS REAL', 'real', confidence, modelUsed, baseNote)
    if (bucket === 'leans-fake') return singleSignal('LEANS FAKE', 'fake', confidence, modelUsed, baseNote)
    // "uncertain" bucket -- no binary call forced.
    return singleSignal(
      'UNCERTAIN', 'amber', confidence, modelUsed,
      'LIAR lands in an uncertain/half-true bucket -- no binary call forced.'
    )
  }

  if (mode === 'factcheck') {
    const fc = raw.fact_check
    if (!fc?.available || fc.confidence == null) {
      return classifierFallback(
        `Web fact-check unavailable (${fc?.reason || 'no result'}) -- showing classifier-only score`
      )
    }
    const baseNote = 'This is the fact-check\u2019s own confidence in this verdict, not a fake/real split.'
    if (fc.verdict === 'TRUE') return singleSignal('TRUE', 'real', fc.confidence, 'web fact-check', baseNote)
    if (fc.verdict === 'FALSE') return singleSignal('FALSE', 'fake', fc.confidence, 'web fact-check', baseNote)
    if (fc.verdict === 'MISLEADING') return singleSignal('MISLEADING', 'amber', fc.confidence, 'web fact-check', baseNote)
    // UNVERIFIED -- no binary call forced.
    return classifierFallback(
      'Web fact-check returned UNVERIFIED (not enough evidence) -- showing classifier-only score'
    )
  }

  // 'combined' mode is rendered entirely by <CombinedPanel /> (its own
  // final-verdict badge, per-model breakdown, and web evidence block) --
  // it doesn't reuse this single-badge/bar layout at all.
  if (mode === 'combined') return null

  return classifierFallback(null)
}

export default function TextInput({ onResult, onError }) {
  const [text, setText] = useState('')
  const [mode, setMode] = useState('classifier')
  const [loading, setLoading] = useState(false)
  // One cached raw API response per mode, so switching tabs restores what
  // that mode last predicted instead of showing another mode's leftover
  // result or going blank.
  const [resultsByMode, setResultsByMode] = useState(EMPTY_RESULTS_BY_MODE)

  const activeMode = MODES.find((m) => m.id === mode)
  const rawResult = resultsByMode[mode]
  const display = deriveDisplay(mode, rawResult)

  const handleRun = async () => {
    const trimmed = text.trim()
    if (!trimmed) {
      onError?.('Type or paste some article text first.')
      return
    }

    setLoading(true)
    onError?.(null)

    try {
      const res = await api.post(activeMode.endpoint, {
        title: '',
        text: trimmed,
      })
      setResultsByMode((prev) => ({ ...prev, [mode]: res.data }))
      onResult?.(res.data)
      onError?.(null)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Prediction failed.'
      onError?.(msg)
      // Only clear this mode's cached result -- leave the other modes'
      // cached data alone so a failed re-run doesn't wipe earlier results.
      setResultsByMode((prev) => ({ ...prev, [mode]: null }))
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setText('')
    setResultsByMode(EMPTY_RESULTS_BY_MODE)
    onError?.(null)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleRun()
    }
  }

  const loadingLabel = {
    classifier: 'Analyzing...',
    ensemble: 'Running ensemble...',
    factcheck: 'Searching...',
    combined: 'Running full ensemble...',
  }[mode]

  return (
    <section className="mb-8">
      <h2 className="font-serif text-xl font-medium mb-3">Text Analysis</h2>
      <div className="rounded-xl border border-[var(--border-strong)] bg-[var(--panel-2)] p-5">
        <p className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)] mb-3">
          Paste a headline or article excerpt
        </p>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Paste a headline and/or a few sentences of article text here..."
          rows={4}
          className="w-full resize-y bg-[var(--bg)] text-[var(--text-primary)] border border-[var(--border-strong)] rounded-lg p-3.5 text-sm leading-relaxed outline-none focus:border-[var(--real)] transition-colors font-[var(--font-body)]"
        />

        {/* Mode selector: choose which backend signal(s) to request. Each
            mode's own last result is cached in resultsByMode, so switching
            tabs here just changes which cached result is displayed -- it
            never re-fetches on its own. */}
        <div className="flex items-center gap-1 mt-3 mb-1 border border-[var(--border)] rounded-lg p-1 bg-[var(--bg)] w-fit flex-wrap">
          {MODES.map((m) => {
            const active = mode === m.id
            return (
              <button
                key={m.id}
                onClick={() => setMode(m.id)}
                className={`font-mono text-[11px] tracking-wide uppercase px-3 py-1.5 rounded-md transition-all ${
                  active
                    ? 'bg-[var(--real)] text-[#0d1a17] font-medium'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }`}
              >
                {m.id === 'factcheck' && <Search className="w-3 h-3 inline mr-1 -mt-0.5" />}
                {m.id === 'ensemble' && <Layers className="w-3 h-3 inline mr-1 -mt-0.5" />}
                {m.id === 'combined' && <Gauge className="w-3 h-3 inline mr-1 -mt-0.5" />}
                {m.label}
              </button>
            )
          })}
        </div>

        <div className="flex items-center gap-3 mt-3">
          <button
            onClick={handleRun}
            disabled={loading || !text.trim()}
            className="font-mono text-xs tracking-wider uppercase px-5 py-2.5 rounded-lg bg-[var(--real)] text-[#0d1a17] font-medium hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-default flex items-center gap-2"
          >
            <Send className="w-3.5 h-3.5" />
            {loading ? loadingLabel : rawResult ? 'Re-run Analysis' : 'Run Analysis'}
          </button>
          <button
            onClick={handleReset}
            disabled={loading}
            className="font-mono text-xs tracking-wider uppercase px-4 py-2.5 rounded-lg border border-[var(--border-strong)] text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors flex items-center gap-2"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
          <span className="font-mono text-[11px] text-[var(--text-muted)] ml-1">
            Ctrl+Enter to run
          </span>
        </div>

        {!display && !rawResult && !loading && (
          <p className="font-mono text-[11px] text-[var(--text-muted)] mt-4">
            No {activeMode.label.toLowerCase()} result yet -- click Run Analysis.
          </p>
        )}

        {/* 'combined' mode has its own self-contained layout (final verdict +
            per-model breakdown + web evidence) -- rendered directly from
            rawResult rather than through the shared badge/bar block below. */}
        {mode === 'combined' && rawResult && (
          <div className="border-t border-[var(--border)] pt-1">
            <CombinedPanel result={rawResult} />
            {rawResult.disclaimer && (
              <p className="font-mono text-[10px] text-[var(--text-muted)] mt-4 leading-relaxed">
                {rawResult.disclaimer}
              </p>
            )}
          </div>
        )}

        {display && (
          <div className="mt-5 border-t border-[var(--border)] pt-5">
            <div className="flex items-center gap-3 mb-4">
              <span
                className={`font-mono text-sm font-medium px-3 py-1 rounded ${TONE_STYLES[display.tone]}`}
              >
                {display.badgeLabel}
              </span>
              <span className="font-mono text-sm text-[var(--text-secondary)]">
                confidence {Math.round(display.confidence * 100)}%
              </span>
              <span className="font-mono text-[10px] text-[var(--text-muted)] uppercase tracking-wide">
                live &middot; {display.modelUsed}
              </span>
            </div>

            {display.hasBinarySplit ? (
              <div className="space-y-2.5">
                <div className="grid grid-cols-[46px_1fr_52px] items-center gap-3">
                  <span className="font-mono text-[11px] text-[var(--text-secondary)]">FAKE</span>
                  <div className="h-2 bg-[var(--bg)] rounded-full overflow-hidden border border-[var(--border)]">
                    <div
                      className="h-full bg-[var(--fake)] rounded-full transition-all duration-600"
                      style={{ width: `${Math.round(display.fakePct * 100)}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs text-[var(--text-secondary)] text-right">
                    {Math.round(display.fakePct * 100)}%
                  </span>
                </div>
                <div className="grid grid-cols-[46px_1fr_52px] items-center gap-3">
                  <span className="font-mono text-[11px] text-[var(--text-secondary)]">REAL</span>
                  <div className="h-2 bg-[var(--bg)] rounded-full overflow-hidden border border-[var(--border)]">
                    <div
                      className="h-full bg-[var(--real)] rounded-full transition-all duration-600"
                      style={{ width: `${Math.round(display.realPct * 100)}%` }}
                    />
                  </div>
                  <span className="font-mono text-xs text-[var(--text-secondary)] text-right">
                    {Math.round(display.realPct * 100)}%
                  </span>
                </div>
              </div>
            ) : (
              // Not a real/fake pair -- a single confidence in one predicted
              // class (see deriveDisplay). One bar, no invented opposite side.
              <div className="grid grid-cols-[46px_1fr_52px] items-center gap-3">
                <span className="font-mono text-[11px] text-[var(--text-secondary)]">CONF</span>
                <div className="h-2 bg-[var(--bg)] rounded-full overflow-hidden border border-[var(--border)]">
                  <div
                    className={`h-full rounded-full transition-all duration-600 ${TONE_BAR_FILL[display.tone]}`}
                    style={{ width: `${Math.round(display.signalPct * 100)}%` }}
                  />
                </div>
                <span className="font-mono text-xs text-[var(--text-secondary)] text-right">
                  {Math.round(display.signalPct * 100)}%
                </span>
              </div>
            )}

            {display.note && (
              <p className="font-mono text-[10px] text-[var(--amber)] mt-3 leading-relaxed">
                {display.note}
              </p>
            )}

            {/* Secondary RoBERTa/LIAR signal, own block -- still the full
                detail behind the derived bar above. */}
            {mode === 'ensemble' && (
              <EnsemblePanel
                liarSignalUsed={rawResult.liar_signal_used}
                liarDetail={rawResult.liar_detail}
                liarGateStatus={rawResult.liar_gate_status}
                unverifiable={rawResult.unverifiable}
              />
            )}

            {/* Agentic fact-check result, own block -- still the full
                detail behind the derived bar above. */}
            {mode === 'factcheck' && <FactCheckPanel factCheck={rawResult.fact_check} />}

            {rawResult.disclaimer && (
              <p className="font-mono text-[10px] text-[var(--text-muted)] mt-4 leading-relaxed">
                {rawResult.disclaimer}
              </p>
            )}
          </div>
        )}
      </div>
    </section>
  )
}