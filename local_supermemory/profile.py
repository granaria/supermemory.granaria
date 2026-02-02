"""Profile Engine - LLM-basierte Profil-Aggregation via Ollama"""
import httpx

PROMPT = """Analysiere folgende Benutzer-Memories und erstelle ein strukturiertes Profil.

MEMORIES:
{memories}

---
Erstelle ein Profil mit: Identität, Hardware/Software, Projekte, Präferenzen, Kontakte.
Nur vorhandene Infos, keine Spekulationen. Deutsch.
"""

class ProfileEngine:
    def __init__(self, url: str = "http://localhost:11434", model: str = "qwen2.5:32b"):
        self.url = url
        self.model = model
    
    async def generate(self, memories: list) -> str:
        if not memories:
            return "Keine Memories."
        
        text = "\n".join([f"- {m.get('content', m)}" for m in memories[:50]])
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as c:
                r = await c.post(f"{self.url}/api/generate", json={
                    "model": self.model,
                    "prompt": PROMPT.format(memories=text),
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 1024}
                })
                if r.status_code == 200:
                    return r.json().get("response", "Fehler")
        except httpx.ConnectError:
            pass
        except Exception as e:
            return f"Fehler: {e}"
        
        # Fallback
        return self._fallback(memories)
    
    def _fallback(self, memories: list) -> str:
        lines = ["## Profil (Fallback)\n"]
        for m in memories[:20]:
            lines.append(f"- {m.get('content', m)}")
        return "\n".join(lines)

_engine = None
def get_engine(model: str = "qwen2.5:32b") -> ProfileEngine:
    global _engine
    if _engine is None or _engine.model != model:
        _engine = ProfileEngine(model=model)
    return _engine
