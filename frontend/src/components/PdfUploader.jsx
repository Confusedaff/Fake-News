import { useState, useRef, useCallback } from 'react'
import { Upload, FileText, AlertCircle, X, Search } from 'lucide-react'
import api from '../api'

const MAX_SIZE_MB = 10
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

export default function PdfUploader({ onResults, onError }) {
  const [state, setState] = useState('idle') // idle | selected | uploading | done | error
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [errorMsg, setErrorMsg] = useState('')
  const [factCheckEnabled, setFactCheckEnabled] = useState(false)
  const inputRef = useRef(null)
  const dropRef = useRef(null)

  const validate = useCallback((f) => {
    if (!f.name.toLowerCase().endsWith('.pdf') && f.type !== 'application/pdf') {
      return 'Only PDF files are accepted.'
    }
    if (f.size > MAX_SIZE_BYTES) {
      return `File too large. Maximum size is ${MAX_SIZE_MB} MB.`
    }
    return null
  }, [])

  const handleFile = useCallback((f) => {
    const err = validate(f)
    if (err) {
      setErrorMsg(err)
      setState('error')
      onError?.(err)
      return
    }
    setFile(f)
    setErrorMsg('')
    setState('selected')
    onError?.(null)
  }, [validate, onError])

  const onDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    dropRef.current?.classList.add('border-[var(--real)]')
  }

  const onDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    dropRef.current?.classList.remove('border-[var(--real)]')
  }

  const onDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    dropRef.current?.classList.remove('border-[var(--real)]')
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }

  const onInputChange = (e) => {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }

  const upload = async () => {
    if (!file) return
    setState('uploading')
    setProgress(0)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await api.post('/analyze-pdf', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        params: { fact_check: factCheckEnabled },
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded / e.total) * 100))
        },
      })
      setState('done')
      onResults?.(res.data)
      onError?.(null)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Upload failed.'
      setErrorMsg(msg)
      setState('error')
      onError?.(msg)
    }
  }

  const reset = () => {
    setFile(null)
    setProgress(0)
    setErrorMsg('')
    setState('idle')
    onResults?.(null)
    onError?.(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <section className="mb-8">
      <h2 className="font-serif text-xl font-medium mb-3">Upload Document</h2>

      <div
        ref={dropRef}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => state === 'idle' && inputRef.current?.click()}
        className={`
          relative rounded-xl border-2 border-dashed p-8 text-center cursor-pointer
          transition-colors duration-200
          ${state === 'idle' ? 'border-[var(--border-strong)] hover:border-[var(--real)] bg-[var(--panel)]' : 'border-[var(--border)] bg-[var(--panel)]'}
          ${state === 'uploading' ? 'cursor-default' : ''}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={onInputChange}
          className="hidden"
        />

        {state === 'idle' && (
          <div className="flex flex-col items-center gap-3">
            <Upload className="w-10 h-10 text-[var(--text-muted)]" />
            <div>
              <p className="text-[var(--text-primary)] font-medium">
                Drop a PDF here or click to browse
              </p>
              <p className="text-sm text-[var(--text-muted)] mt-1">
                Maximum file size: {MAX_SIZE_MB} MB
              </p>
            </div>
          </div>
        )}

        {(state === 'selected' || state === 'uploading') && (
          <div className="flex flex-col items-center gap-3">
            <FileText className="w-10 h-10 text-[var(--real)]" />
            <div>
              <p className="text-[var(--text-primary)] font-medium">{file?.name}</p>
              <p className="text-sm text-[var(--text-muted)]">{formatSize(file?.size || 0)}</p>
            </div>

            {state === 'uploading' && (
              <div className="w-full max-w-xs mt-2">
                <div className="h-2 bg-[var(--bg)] rounded-full overflow-hidden border border-[var(--border)]">
                  <div
                    className="h-full bg-[var(--real)] rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="font-mono text-xs text-[var(--text-muted)] mt-1 text-center">
                  Uploading... {progress}%
                </p>
              </div>
            )}

            {state === 'selected' && (
              <>
                <label className="flex items-center gap-2 mt-1 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={factCheckEnabled}
                    onChange={(e) => setFactCheckEnabled(e.target.checked)}
                    className="w-3.5 h-3.5 accent-[var(--real)]"
                  />
                  <span className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wide text-[var(--text-secondary)]">
                    <Search className="w-3 h-3" />
                    Real-world fact-check (unsupported/uncertain segments, slower)
                  </span>
                </label>
                <div className="flex gap-2 mt-2">
                  <button
                    onClick={upload}
                    className="font-mono text-xs tracking-wider uppercase px-5 py-2.5 rounded-lg bg-[var(--real)] text-[#0d1a17] font-medium hover:opacity-90 transition-opacity"
                  >
                    Analyze Document
                  </button>
                  <button
                    onClick={reset}
                    className="font-mono text-xs tracking-wider uppercase px-4 py-2.5 rounded-lg border border-[var(--border-strong)] text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {state === 'done' && (
          <div className="flex flex-col items-center gap-3">
            <FileText className="w-10 h-10 text-[var(--real)]" />
            <div>
              <p className="text-[var(--text-primary)] font-medium">{file?.name}</p>
              <p className="text-sm text-[var(--real)]">Analysis complete</p>
            </div>
            <button
              onClick={reset}
              className="font-mono text-xs tracking-wider uppercase px-4 py-2 rounded-lg border border-[var(--border-strong)] text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors"
            >
              Upload Another
            </button>
          </div>
        )}

        {state === 'error' && (
          <div className="flex flex-col items-center gap-3">
            <AlertCircle className="w-10 h-10 text-[var(--fake)]" />
            <div>
              <p className="text-[var(--fake)] font-medium">{errorMsg}</p>
            </div>
            <button
              onClick={reset}
              className="font-mono text-xs tracking-wider uppercase px-4 py-2 rounded-lg border border-[var(--border-strong)] text-[var(--text-secondary)] hover:border-[var(--text-muted)] transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </section>
  )
}