"""RAG Engine — Query Expansion + Context-aware Answer Generation.

Portiert aus supermemoryai/supermemory v2/apps/cf-ai-backend/src/prompts/prompt1.ts
— dort mit <context> / <context_score> / <justification> / <answer> Tags und
englischem System-Prompt. Hier: deutsches Pendant, Ollama statt Cloudflare AI,
mit robuster Fallback-Kette für den Offline-Fall.

Komponenten:

* QueryExpander: paraphrasiert eine Suchanfrage in N alternative
  Formulierungen (synonyme Begriffe, andere Schlüsselwörter) — fängt
  Mehrdeutigkeit und Synonymie ab, die ein einzelnes Embedding nicht löst.

* RAGEngine: baut den v2-Context-Prompt mit similarity- und
  normalised_score-Tags, ruft Ollama auf und parst die Antwort in
  justification + answer. Bei Offline-Ollama: textuelle Fallback-Zusammen-
  fassung der Top-Memories.

* recall_multi(): top-level Orchestrierung — Query-Expansion + parallele
  recalls + memory_id-Dedup + Score-Re-Normalisierung.
"""
from __future__ import annotations
import asyncio
import re
from typing import Optional

import httpx


# ═══════════════════════════════════════════════════════════════
# Query Expansion
# ═══════════════════════════════════════════════════════════════

QUERY_EXPANSION_PROMPT = """Du hilfst bei einer semantischen Wissensdatenbank-Suche.
Gegeben ist die folgende Suchanfrage. Generiere {n} alternative Formulierungen
(Paraphrasen, Synonyme, verwandte Fachbegriffe), die die gleiche Information
abrufen würden.

Regeln:
- Eine Formulierung pro Zeile.
- Keine Nummerierung, keine Erklärung, keine Einleitung.
- Behalte die Sprache des Originals bei.
- Keine triviale Umformulierung — echte Alternativen mit anderen Keywords.
- Keine Vermutungen über den Kontext.

Original: {query}

Alternativen:"""


class QueryExpander:
    """Paraphrasiert Suchanfragen via Ollama für Multi-Query Retrieval."""

    def __init__(self, url: str = "http://localhost:11434",
                 model: str = "qwen2.5:32b"):
        self.url = url
        self.model = model

    async def expand(self, query: str, n: int = 3) -> list[str]:
        """Return [original] + bis zu n Paraphrasen. Fallback: nur [original]."""
        if n < 1 or not query.strip():
            return [query]

        try:
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(f"{self.url}/api/generate", json={
                    "model": self.model,
                    "prompt": QUERY_EXPANSION_PROMPT.format(query=query, n=n),
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 256,
                    }
                })
                if r.status_code != 200:
                    return [query]

                text = r.json().get("response", "").strip()
                # Parse zeilenweise, Nummerierungen strippen
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                lines = [re.sub(r"^[\-\*\d]+[\.\)]?\s*", "", l) for l in lines]

                # Dedup case-insensitive, Original ausschließen
                seen = {query.strip().lower()}
                alternatives = []
                for l in lines:
                    key = l.lower()
                    if key in seen or len(l) < 3:
                        continue
                    seen.add(key)
                    alternatives.append(l)
                return [query] + alternatives[:n]
        except Exception:
            return [query]


# ═══════════════════════════════════════════════════════════════
# Multi-Query Recall (top-level orchestration)
# ═══════════════════════════════════════════════════════════════

async def recall_multi(store, query: str, project: str = "default",
                       n: int = 15, expand_n: int = 3) -> dict:
    """Query-Expansion + parallele Recalls + Merge + Re-Normalisierung.

    Args:
        store: MemoryStore-Instanz.
        query: User-Suchanfrage.
        project: Projekt-Namespace.
        n: Max. Anzahl finaler Memories.
        expand_n: Anzahl Paraphrasen (0 = kein Expansion = normales recall).

    Returns:
        {"memories": [...], "queries": [original, paraphrase1, ...]}

    Merging-Strategie:
        Pro memory_id wird der höchste similarity-Score über alle Query-
        Varianten behalten. `matched_query` im Output zeigt, welche Query
        diesen Treffer produziert hat — diagnostisch wertvoll.
    """
    expander = get_expander()
    queries = await expander.expand(query, n=expand_n) if expand_n > 0 else [query]

    # store.recall ist synchron — parallel via to_thread, damit die
    # Queries wirklich nebeneinander laufen
    recall_calls = [
        asyncio.to_thread(store.recall, q, project, n)
        for q in queries
    ]
    all_results = await asyncio.gather(*recall_calls, return_exceptions=True)

    # Merge: pro memory_id den besten Hit behalten
    best: dict[str, dict] = {}
    for q, result in zip(queries, all_results):
        if isinstance(result, Exception):
            continue
        for hit in result:
            mid = hit["id"]
            if mid not in best or hit["similarity"] > best[mid]["similarity"]:
                h = dict(hit)
                h["matched_query"] = q
                best[mid] = h

    merged = sorted(best.values(), key=lambda x: -x["similarity"])[:n]

    # Score-Re-Normalisierung über das Merged-Set
    if merged:
        sims = [h["similarity"] for h in merged]
        lo, hi = min(sims), max(sims)
        for h in merged:
            if hi == lo:
                h["normalised_score"] = 50
            else:
                h["normalised_score"] = round(
                    ((h["similarity"] - lo) / (hi - lo)) * 99 + 1
                )

    return {"memories": merged, "queries": queries}


# ═══════════════════════════════════════════════════════════════
# RAG Answer Generation
# ═══════════════════════════════════════════════════════════════

RAG_SYSTEM_PROMPT = """Du bist ein Second-Brain-Assistent, der Fragen des Nutzers auf Basis des bereitgestellten Kontexts beantwortet. Ziel: die Frage direkt und präzise beantworten, ohne unnötige Ausführungen.

Der Kontext besteht aus mehreren Memories mit Ähnlichkeits-Scores. Jede Memory steht in <context>-Tags, die Scores in <context_score>-Tags. Die Frage steht in <question>-Tags.

Anleitung:
- Analysiere die Frage, identifiziere die Kerninformation.
- Lokalisiere die relevanten Passagen in den Kontexten.
- Vergleiche die Scores: normalised_score (1..100) ist relativ im Resultset (100 = bester Match); similarity (0..100) ist absolute Cosine-Similarity.
- Schreibe in <justification>-Tags eine kurze Begründung (1–2 Sätze), welche Kontexte am relevantesten sind.
- Fasse die Info in eigenen Worten zusammen und gib eine direkte Antwort in <answer>-Tags.
- Verwende Markdown (fett, kursiv, Aufzählungen) für Lesbarkeit.
- Bei unpassendem/fehlendem Kontext: sag, dass du nichts Passendes gespeichert hast. Keine Spekulationen.

Antworte auf Deutsch. Strikt dieses Format:

<justification>
...
</justification>
<answer>
...
</answer>"""


RAG_USER_TEMPLATE = """{contexts}

<question>
{question}
</question>"""


def build_contexts_block(memories: list[dict]) -> str:
    """Serialisiert Memories in v2-Context-Blöcke.

    Nutzt matched_chunk (falls vorhanden), sonst den Volltext. Titel wird
    als Prefix in eckigen Klammern ergänzt.
    """
    parts = []
    for m in memories:
        content = m.get("matched_chunk") or m.get("content", "")
        sim = m.get("similarity", 0)
        norm = m.get("normalised_score", 50)
        prefix = f"[{m['title']}] " if m.get("title") else ""
        parts.append(
            f"<context>\n{prefix}{content}\n</context>\n"
            f"<context_score>\nsimilarity: {sim}\n"
            f"normalised_score: {norm}\n</context_score>"
        )
    return "\n".join(parts)


class RAGEngine:
    """Generiert RAG-Antworten via Ollama im v2-Tag-Format."""

    def __init__(self, url: str = "http://localhost:11434",
                 model: str = "qwen2.5:32b"):
        self.url = url
        self.model = model

    async def answer(self, query: str, memories: list[dict],
                     max_tokens: int = 1500) -> dict:
        """Return {answer, justification, provider[, error]}.

        Fallback-Kette:
        1. Normale Antwort via Ollama mit v2-Prompt.
        2. Bei Parse-Fehler: ganze Ollama-Antwort = answer, generische
           justification.
        3. Bei Ollama-Fehler: textueller Summary der Top-Memories.
        """
        if not memories:
            return {
                "answer": "Ich habe nichts Passendes in meinen Memories gefunden.",
                "justification": "Retrieval-Schritt lieferte keine Treffer.",
                "provider": "none"
            }

        contexts = build_contexts_block(memories)
        user_prompt = RAG_USER_TEMPLATE.format(
            contexts=contexts, question=query
        )
        full_prompt = f"{RAG_SYSTEM_PROMPT}\n\n{user_prompt}"

        try:
            async with httpx.AsyncClient(timeout=180.0) as c:
                r = await c.post(f"{self.url}/api/generate", json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": max_tokens,
                    }
                })
                if r.status_code == 200:
                    response = r.json().get("response", "").strip()
                    parsed = self._parse_response(response)
                    parsed["provider"] = f"Ollama ({self.model})"
                    return parsed
                else:
                    return {
                        "answer": self._fallback_answer(memories),
                        "justification": f"Ollama HTTP {r.status_code}.",
                        "provider": "fallback",
                        "error": f"HTTP {r.status_code}"
                    }
        except httpx.ConnectError:
            return {
                "answer": self._fallback_answer(memories),
                "justification": "Ollama nicht erreichbar — reine Retrieval-Ergebnisse.",
                "provider": "fallback"
            }
        except Exception as e:
            return {
                "answer": self._fallback_answer(memories),
                "justification": f"LLM-Fehler: {type(e).__name__}.",
                "provider": "fallback",
                "error": str(e)
            }

    def _parse_response(self, text: str) -> dict:
        """Extrahiere <justification> und <answer> Tags."""
        just_match = re.search(r"<justification>(.*?)</justification>",
                               text, re.DOTALL | re.IGNORECASE)
        ans_match = re.search(r"<answer>(.*?)</answer>",
                              text, re.DOTALL | re.IGNORECASE)

        if just_match and ans_match:
            return {
                "justification": just_match.group(1).strip(),
                "answer": ans_match.group(1).strip()
            }

        # Teil-Match: nur answer vorhanden
        if ans_match:
            return {
                "justification": "(Modell lieferte keine explizite Begründung)",
                "answer": ans_match.group(1).strip()
            }

        # Kein Tag — vermutlich hat das Modell das Format ignoriert.
        # Trotzdem nützlich, als ganze Antwort zurückgeben.
        return {
            "justification": "(Modell antwortete ohne strukturiertes Format)",
            "answer": text
        }

    def _fallback_answer(self, memories: list[dict]) -> str:
        """Plain-Text-Zusammenfassung der Top-Memories ohne LLM."""
        if not memories:
            return "Keine Memories gefunden."
        lines = ["Basierend auf folgenden Memories (reines Retrieval, ohne LLM-Synthese):\n"]
        for i, m in enumerate(memories[:5], 1):
            content = m.get("matched_chunk") or m.get("content", "")
            snippet = content[:300] + ("…" if len(content) > 300 else "")
            title = f" — **{m['title']}**" if m.get("title") else ""
            lines.append(f"**{i}.** *(sim {m.get('similarity', 0)}%)*{title}\n{snippet}")
        return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Singletons
# ═══════════════════════════════════════════════════════════════

_expander: Optional[QueryExpander] = None
_rag_engine: Optional[RAGEngine] = None


def get_expander(model: str = "qwen2.5:32b") -> QueryExpander:
    global _expander
    if _expander is None or _expander.model != model:
        _expander = QueryExpander(model=model)
    return _expander


def get_rag_engine(model: str = "qwen2.5:32b") -> RAGEngine:
    global _rag_engine
    if _rag_engine is None or _rag_engine.model != model:
        _rag_engine = RAGEngine(model=model)
    return _rag_engine
