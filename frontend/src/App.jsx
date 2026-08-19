import { useState } from 'react'
import { FileText, Type } from 'lucide-react'
import Header from './components/Header'
import TextInput from './components/TextInput'
import PdfUploader from './components/PdfUploader'
import ResultsTable from './components/ResultsTable'
import SummaryStats from './components/SummaryStats'

const TABS = [
  { id: 'text', label: 'Text Analysis', icon: Type },
  { id: 'pdf', label: 'PDF Upload', icon: FileText },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('text')
  const [pdfResults, setPdfResults] = useState(null)
  const [pdfError, setPdfError] = useState(null)
  const [textError, setTextError] = useState(null)

  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <Header />
      <main className="max-w-5xl mx-auto px-6 py-8">
        {/* Tabs */}
        <div className="flex gap-1 mb-6 border border-[var(--border)] rounded-lg p-1 bg-[var(--panel)] w-fit">
          {TABS.map((tab) => {
            const Icon = tab.icon
            const active = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-2 font-mono text-xs tracking-wider uppercase px-5 py-2.5 rounded-md transition-all
                  ${active
                    ? 'bg-[var(--real)] text-[#0d1a17] font-medium'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                  }
                `}
              >
                <Icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Tab content -- both tabs stay mounted at all times (just
            hidden via CSS) so switching tabs never unmounts TextInput or
            PdfUploader. If we conditionally rendered them out of the
            tree instead, React would throw away all their internal
            useState (typed text, cached results per mode, uploaded PDF
            results) every time you switched away and back. */}
        <div className={activeTab === 'text' ? '' : 'hidden'}>
          <TextInput
            onResult={() => setTextError(null)}
            onError={setTextError}
          />
        </div>

        <div className={activeTab === 'pdf' ? '' : 'hidden'}>
          <PdfUploader
            onResults={setPdfResults}
            onError={setPdfError}
          />

          {pdfError && !pdfResults && (
            <div className="mb-6 p-4 rounded-lg bg-[var(--fake-dim)] border border-[var(--fake)]">
              <p className="text-sm text-[var(--fake)]">{pdfError}</p>
            </div>
          )}

          {pdfResults && (
            <>
              <SummaryStats summary={pdfResults.summary} />
              <ResultsTable segments={pdfResults.segments} />

              {pdfResults.disclaimer && (
                <div className="mb-6 p-3 rounded-lg bg-[var(--panel-2)] border border-[var(--border)]">
                  <p className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] mb-1">API Disclaimer</p>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">{pdfResults.disclaimer}</p>
                </div>
              )}
            </>
          )}
        </div>

      </main>

      <footer className="border-t border-[var(--border)] mt-12">
        <div className="max-w-5xl mx-auto px-6 py-5 font-mono text-[11px] text-[var(--text-muted)] space-y-1">
          <div><strong className="text-[var(--text-secondary)]">Data</strong> Kaggle Fake and Real News Dataset &middot; ~44,900 articles</div>
          <div><strong className="text-[var(--text-secondary)]">Backend</strong> FastAPI + scikit-learn Linear SVM + PyPDF2</div>
          <div><strong className="text-[var(--text-secondary)]">Ensemble</strong> RoBERTa/LIAR style signal (gated, secondary) + Groq web-search fact-check (agentic, real-world)</div>
          <div><strong className="text-[var(--text-secondary)]">Full ensemble</strong> combines all three signals into one final verdict via a weighted blend -- web fact-check weighted highest (60%), TF-IDF 25%, LIAR 15%</div>
          <div><strong className="text-[var(--text-secondary)]">Frontend</strong> React + Vite + Tailwind CSS</div>
        </div>
      </footer>
    </div>
  )
}
