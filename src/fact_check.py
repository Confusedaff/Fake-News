"""
Agentic real-world fact-checking, layered on top of (never replacing) the
TF-IDF style classifier.

Why this exists
----------------
The TF-IDF model (see src/preprocessing.py, docs/limitations.md) answers
"does this text's style resemble known credible/non-credible sources,"
which is NOT the same question as "is this claim true right now." A short,
casually-phrased, factually-true statement like "donald trump is the
president of USA" can score confidently "fake" on style grounds alone --
the model has no way to check facts, because it was never given anything
to check them against.

This module gives the system that missing capability, using a
search-then-generate design rather than an agentic tool-use loop:

  1. Search step (cheap, tightly-bounded LLM tokens): query Wikipedia's
     free, keyless OpenSearch endpoint for an authoritative reference,
     and groq/compound-mini -- restricted, via compound_custom, to ONLY
     its web_search tool (no code_interpreter, no visit_website) -- for
     a couple of recent articles.
  2. Generate step: hand the claim plus a small, hard-capped block of
     already-fetched snippets to a PLAIN Groq chat model with no tools
     attached at all, and ask it to return a structured verdict.

Why groq/compound-mini, restricted to one tool, for the search step
---------------------------------------------------------------------
Groq's standalone `web_search` tool (the one that returns clean
{title, url, content} results via message.executed_tools[].search_results)
is only ever available on the compound systems (groq/compound,
groq/compound-mini) -- it cannot be attached to a plain chat model via
the "tools" param the way this module originally tried to. Plain chat
models (e.g. the openai/gpt-oss-* family) only support the *different*
`browser_search` tool, which drives an interactive multi-page browsing
session and returns an unstructured cited answer, not a snippet list --
not a fit for this module's "give me a couple of named sources" design.

Compound systems do run their own server-side agentic loop, so their
token cost isn't fully bounded from the outside the way a plain
non-agentic call is. compound_custom.tools.enabled_tools=["web_search"]
narrows that loop to a single tool (no code_interpreter, no
visit_website), which is the closest available bound; MAX_TOKENS caps
the final synthesis length on top of that.

By hard-capping how many characters of snippet text we forward into the
*generation* step, that request's size stays fully deterministic and
under our control regardless of what the search step returns. This also
directly answers the request for named reference sources (Wikipedia plus
a couple of recent articles) rather than an opaque black-box search.

Design choices, mirroring src/liar_ensemble.py's gating pattern
------------------------------------------------------------------
- This is an ADDITIONAL signal, never a silent override. The TF-IDF label
  and confidence are always returned unchanged alongside this result, the
  same way liar_ensemble.py never lets the LIAR signal overwrite the
  TF-IDF label.
- Fails safe. If the API key is missing, a step times out, or the
  response can't be parsed, this returns a result with available=False
  and a human-readable reason -- never a 500, never a fabricated verdict.
- Deferred client construction, so a deployment without a GROQ_API_KEY
  set never pays any cost or throws at import time.
- Bounded: two small HTTP calls (Wikipedia + one narrow Groq web_search
  call) with hard timeouts, then one plain generation call with no tools
  attached and a hard-capped prompt size.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Plain, non-agentic model for the final verdict -- no compound system, no
# tool-use loop, so the request's token size is exactly what we send it,
# nothing more.
# NOTE: llama-3.3-70b-versatile was deprecated by Groq (announced
# 2026-06-17) and now 404s. openai/gpt-oss-120b is Groq's recommended
# migration target.
GENERATION_MODEL = "openai/gpt-oss-120b"
# groq/compound-mini, restricted (below, via compound_custom) to ONLY its
# web_search tool, used to fetch a couple of recent-article snippets. The
# standalone web_search tool only exists on the compound systems -- it is
# NOT a "tools": [...] option that can be attached to a plain chat model
# (that combination silently fails and was why search results were never
# coming back).
SEARCH_MODEL = "groq/compound-mini"

MAX_TOKENS = 768
REQUEST_TIMEOUT_S = 20
# Hard caps on everything we forward into the generation prompt, so its
# size is fully deterministic regardless of how much text a search step
# returns.
MAX_CLAIM_CHARS = 500
MAX_SNIPPET_CHARS = 400   # per search result, after trimming
MAX_SNIPPETS = 3          # 1 Wikipedia + up to 2 recent articles

VERDICTS = {"TRUE", "FALSE", "MISLEADING", "UNVERIFIED"}

GENERATION_SYSTEM_PROMPT = """You are a fact-checking assistant. You will be given a claim and a
short list of already-fetched reference snippets (title, URL, short excerpt). Do not search the
web yourself -- base your verdict only on the snippets provided.

Rules:
- Base your verdict only on the snippets given to you. If they don't cover the claim, say so.
- If snippets conflict or are insufficient, say so honestly -- do not guess.
- In "sources", return the URLs from the snippets you actually used (at most 3).
- Respond with ONLY a JSON object (no markdown fences, no preamble) matching exactly:
  {
    "verdict": "TRUE" | "FALSE" | "MISLEADING" | "UNVERIFIED",
    "confidence": <float 0.0-1.0>,
    "explanation": "<2-3 sentences, plain language, citing what the snippets said>",
    "sources": ["<url1>", "<url2>", "<url3>"]
  }
- "MISLEADING" means technically-true-but-deceptively-framed, or true-with-important-missing-context.
- "UNVERIFIED" means the snippets don't give enough information either way -- this is a legitimate
  answer, not a failure. Never force TRUE or FALSE without snippet support.
- Keep the explanation brief -- 2-3 sentences, no extra commentary outside the JSON.
"""


@dataclass
class FactCheckResult:
    available: bool
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    sources: list = field(default_factory=list)
    reason: Optional[str] = None  # why unavailable, or a note when available


@dataclass
class Snippet:
    title: str
    url: str
    text: str


class FactChecker:
    """
    search-then-generate fact-checker: Wikipedia + Groq's standalone
    web_search tool for retrieval (cheap, no generation tokens), then one
    plain Groq chat completion (no tools) for the verdict. The API key is
    just read from the environment at call time, so a deployment with no
    GROQ_API_KEY set never throws at import time.
    """

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.available = bool(self.api_key)
        self.reason = (
            "ready" if self.available
            else "GROQ_API_KEY not set -- agentic fact-check disabled, "
                 "falling back to classifier-only results"
        )

    def check(self, claim: str) -> FactCheckResult:
        if not self.available:
            return FactCheckResult(available=False, reason=self.reason)

        claim = (claim or "").strip()
        if not claim:
            return FactCheckResult(available=False, reason="Empty claim text.")

        truncated = len(claim) > MAX_CLAIM_CHARS
        if truncated:
            claim = claim[:MAX_CLAIM_CHARS]

        try:
            snippets = self._gather_snippets(claim)
        except Exception as exc:
            # Retrieval failures fail soft: fall through with whatever we
            # have (possibly empty), and let the generation step return
            # UNVERIFIED rather than erroring the whole request out.
            snippets = []

        try:
            return self._generate_verdict(claim, snippets, truncated=truncated)
        except requests.exceptions.Timeout:
            return FactCheckResult(
                available=False,
                reason=f"Fact-check call timed out after {REQUEST_TIMEOUT_S}s; "
                       f"falling back to classifier-only result",
            )
        except Exception as exc:
            return FactCheckResult(
                available=False,
                reason=f"Fact-check call failed ({type(exc).__name__}: {exc}); "
                       f"falling back to classifier-only result",
            )

    # ---- retrieval (no generation tokens spent) ----

    def _gather_snippets(self, claim: str) -> list:
        snippets = []

        wiki = self._search_wikipedia(claim)
        if wiki:
            snippets.append(wiki)

        try:
            articles = self._search_recent_articles(claim, limit=MAX_SNIPPETS - len(snippets))
            snippets.extend(articles)
        except Exception:
            pass  # recent-article search is best-effort; Wikipedia alone is still useful

        return snippets[:MAX_SNIPPETS]

    def _search_wikipedia(self, claim: str) -> Optional[Snippet]:
        """Free, keyless Wikipedia search -- the authoritative-reference half
        of the source list. No LLM tokens involved."""
        try:
            resp = requests.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": claim,
                    "srlimit": 1,
                    "format": "json",
                    "origin": "*",
                },
                headers={"User-Agent": "FakeNewsDetection1-FactChecker/1.0"},
                timeout=REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
            results = (resp.json().get("query", {}) or {}).get("search", [])
            if not results:
                return None
            top = results[0]
            title = top.get("title", "")
            # Strip HTML tags MediaWiki includes in the snippet (e.g. <span class="searchmatch">)
            import re
            raw_snippet = re.sub(r"<[^>]+>", "", top.get("snippet", ""))
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            return Snippet(title=title, url=url, text=raw_snippet[:MAX_SNIPPET_CHARS])
        except Exception:
            return None

    def _search_recent_articles(self, claim: str, limit: int) -> list:
        """groq/compound-mini, restricted to its web_search tool only, used
        to fetch snippets -- the model's own prose response is discarded,
        we only read executed_tools[].search_results.results. Restricting
        via compound_custom (rather than the OpenAI-style "tools" param,
        which compound systems don't use to opt into web_search) keeps the
        search step narrow and its cost as separate from the generation
        step as compound's agentic loop allows."""
        if limit <= 0:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": SEARCH_MODEL,
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": f"Find recent news articles about: {claim}"}
            ],
            "compound_custom": {
                "tools": {"enabled_tools": ["web_search"]}
            },
        }
        resp = requests.post(
            GROQ_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S
        )
        if resp.status_code != 200:
            return []  # best-effort -- Wikipedia snippet alone is still useful

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return []

        message = choices[0].get("message", {})
        executed_tools = message.get("executed_tools") or []
        results = []
        for tool in executed_tools:
            # search_results is {"results": [{"title", "url", "content", "score"}, ...]},
            # not a bare list -- indexing it directly (the earlier bug here)
            # silently iterated the dict's keys instead of its results.
            search_results = tool.get("search_results") or {}
            for r in (search_results.get("results") or [])[:limit]:
                title = r.get("title", "")
                url = r.get("url", "")
                text = (r.get("content") or r.get("snippet") or "")[:MAX_SNIPPET_CHARS]
                if url:
                    results.append(Snippet(title=title, url=url, text=text))
            if results:
                break
        return results[:limit]

    # ---- generation (the only step that spends meaningful tokens) ----

    def _generate_verdict(self, claim: str, snippets: list, truncated: bool) -> FactCheckResult:
        if not snippets:
            snippet_block = "(No reference snippets were found for this claim.)"
        else:
            lines = []
            for s in snippets:
                lines.append(f"- {s.title} ({s.url}): {s.text}")
            snippet_block = "\n".join(lines)

        user_content = f'Claim: "{claim}"\n\nReference snippets:\n{snippet_block}'

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GENERATION_MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            # openai/gpt-oss-120b is a reasoning model: by default it burns an
            # unpredictable share of max_tokens on hidden reasoning before it
            # ever starts writing the JSON, which was cutting the response
            # off mid-object and causing "Could not parse a JSON verdict"
            # errors. "low" keeps reasoning short for this simple
            # extraction/classification task, leaving the budget for the
            # actual answer.
            "reasoning_effort": "low",
            # JSON mode: Groq/OpenAI-compatible constrained decoding that
            # guarantees the emitted message content is syntactically valid
            # JSON (it does not enforce our exact schema, so the
            # VERDICTS/parsed.get(...) checks below still matter). Requires
            # the word "json" to appear in the prompt, which
            # GENERATION_SYSTEM_PROMPT already does.
            "response_format": {"type": "json_object"},
            # Deliberately NO "tools" key -- this is a plain generation
            # call, not an agentic one. Nothing server-side can inflate
            # this request's token cost beyond what we sent it.
        }

        resp = requests.post(
            GROQ_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S
        )

        if resp.status_code == 401:
            return FactCheckResult(
                available=False,
                reason="Groq API rejected the key (401) -- check GROQ_API_KEY; "
                       "falling back to classifier-only result",
            )
        if resp.status_code == 413:
            detail = ""
            try:
                detail = (resp.json().get("error", {}) or {}).get("message", "")
            except Exception:
                pass
            return FactCheckResult(
                available=False,
                reason="Groq API 413 -- request exceeded the tokens-per-minute "
                       "limit for your tier, even on a plain non-agentic call. "
                       "Consider lowering MAX_TOKENS/MAX_SNIPPET_CHARS further "
                       "if this recurs on the free tier."
                       + (f": {detail}" if detail else "")
                       + ". Falling back to classifier-only result.",
            )
        if resp.status_code == 429:
            return FactCheckResult(
                available=False,
                reason="Groq API rate limit hit (429) -- falling back to "
                       "classifier-only result",
            )
        resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return FactCheckResult(
                available=False,
                reason="Groq response contained no choices; falling back to "
                       "classifier-only result",
            )

        finish_reason = choices[0].get("finish_reason")
        raw_text = (choices[0].get("message", {}).get("content") or "").strip()
        if not raw_text:
            hint = (
                " (finish_reason=length -- the model likely spent its whole "
                "token budget on reasoning before writing any content; try "
                "raising MAX_TOKENS or lowering reasoning_effort further)"
                if finish_reason == "length" else ""
            )
            return FactCheckResult(
                available=False,
                reason=f"Model returned no text content{hint}; falling back "
                       f"to classifier-only result",
            )

        parsed = self._parse_verdict_json(raw_text)
        if parsed is None:
            hint = (
                " (finish_reason=length -- response was likely truncated "
                "mid-JSON; try raising MAX_TOKENS)"
                if finish_reason == "length" else ""
            )
            return FactCheckResult(
                available=False,
                reason=f"Could not parse a JSON verdict from the model "
                       f"response{hint}; falling back to classifier-only "
                       f"result",
            )

        verdict = parsed.get("verdict")
        if verdict not in VERDICTS:
            return FactCheckResult(
                available=False,
                reason=f"Model returned an unrecognized verdict ({verdict!r}); "
                       f"falling back to classifier-only result",
            )

        confidence = parsed.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = None

        # Prefer the model's own source list, but fall back to the URLs we
        # actually fetched if the model returned none -- we know these are
        # real, since we fetched them ourselves.
        sources = parsed.get("sources") or [s.url for s in snippets]

        note_parts = []
        if not snippets:
            note_parts.append("no reference snippets were found; verdict is likely UNVERIFIED")
        if truncated:
            note_parts.append(f"claim text truncated to {MAX_CLAIM_CHARS} chars before sending")
        note = "ok" if not note_parts else "ok (" + "; ".join(note_parts) + ")"

        return FactCheckResult(
            available=True,
            verdict=verdict,
            confidence=confidence,
            explanation=parsed.get("explanation"),
            sources=sources,
            reason=note,
        )

    @staticmethod
    def _parse_verdict_json(raw_text: str) -> Optional[dict]:
        """Best-effort JSON extraction: the prompt asks for JSON-only, but
        models sometimes wrap it in a code fence or add a stray sentence.
        This does not attempt to salvage garbage -- if it can't find a
        clean JSON object, it returns None and the caller fails safe."""
        candidate = raw_text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
            candidate = candidate.strip()

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None

        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None


# Module-level singleton -- construction is cheap (just checks an env var);
# the actual HTTP calls happen lazily on first .check() call.
fact_checker = FactChecker()