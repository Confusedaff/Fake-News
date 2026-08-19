export default function Header() {
  return (
    <header className="border-b border-[var(--border)]">
      <div className="max-w-5xl mx-auto px-6 py-10">
        <p className="font-mono text-xs tracking-[0.12em] uppercase text-[var(--amber)] mb-3">
          Document Claim-Support Assessment
        </p>
        <h1 className="font-serif text-4xl md:text-5xl font-medium leading-tight">
          Fake News Detection<br />
          <span className="text-[var(--real)]">Claim Analyzer</span>
        </h1>
      </div>
    </header>
  )
}
