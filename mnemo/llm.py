"""A large language model next to the mind: Mnemo answers what it knows,
the model what it does not.

Mnemo is exact, explained, instant and small, but it knows only what it was
told or fed, and it reads a controlled English.  A language model is fluent
and broad, but slow, opaque and sometimes wrong.  Paired:

  1. Mnemo answers first, from memory (well under a millisecond, with reasons);
  2. if it does not know, or does not understand, the model answers, shown
     the facts Mnemo holds about the words of the question (retrieval from a
     symbolic memory, so that the model is grounded in it);
  3. the answer carries the model's name, and 'why?' says where it came from;
  4. a short answer that Mnemo can read is kept as a fact (source 'llm', low
     confidence), unless it contradicts what Mnemo firmly believes; next time
     Mnemo answers from memory.

Backends (standard library only; nothing is downloaded, the model runs on
your machine or your own server):

  ollama:MODEL[@http://host:11434]   Ollama's /api/chat ('ollama:gemma:2b')
  openai:MODEL[@http://host:8080/v1] any OpenAI-compatible server: llama.cpp's
                                     llama-server, vLLM, LM Studio, ...
"""
import json
import re
import urllib.parse
import urllib.request

from . import language as L

SYSTEM = ("You are the language part of Mnemo, a small reasoning system. Answer in one or "
          "two short, plain sentences. For a yes/no question, begin with Yes or No. If you "
          "do not know, say \"I don't know.\" Do not make facts up.")

_DEFAULT = {"ollama": "http://localhost:11434", "openai": "http://localhost:8080/v1"}


class LLM:
    """A chat model behind an HTTP API.  spec: 'ollama:gemma:2b',
    'openai:qwen2.5-0.5b-instruct@http://127.0.0.1:8080/v1'."""

    def __init__(self, spec, timeout=300, max_tokens=160):
        backend, _, rest = spec.partition(":")
        if backend not in _DEFAULT or not rest:
            raise ValueError(f"model spec must be ollama:MODEL or openai:MODEL[@URL], not {spec!r}")
        model, _, endpoint = rest.partition("@")
        self.backend, self.model = backend, model
        self.endpoint = (endpoint or _DEFAULT[backend]).rstrip("/")
        self.timeout, self.max_tokens = timeout, max_tokens
        host = urllib.parse.urlparse(self.endpoint).hostname or ""
        # a model on this machine is reached directly, never through a proxy
        local = host in ("localhost", "127.0.0.1", "::1")
        self._open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open \
            if local else urllib.request.urlopen

    @property
    def name(self):
        return self.model

    def ask(self, prompt, system=SYSTEM, history=()):
        """The model's reply to prompt; history: earlier (question, answer)
        pairs of the conversation."""
        msgs = [{"role": "system", "content": system}] if system else []
        for q, a in history:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": prompt})
        if self.backend == "ollama":
            url = self.endpoint + "/api/chat"
            body = {"model": self.model, "messages": msgs, "stream": False,
                    "options": {"temperature": 0, "num_predict": self.max_tokens}}
        else:
            url = self.endpoint + "/chat/completions"
            body = {"model": self.model, "messages": msgs, "temperature": 0,
                    "max_tokens": self.max_tokens}
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with self._open(req, timeout=self.timeout) as r:
            out = json.load(r)
        text = out["message"]["content"] if self.backend == "ollama" else \
            out["choices"][0]["message"]["content"]
        return text.strip()


# ---------------------------------------------------------------- what to show it

_STOP = set("""a an the is are was were be been being of in on at to for from by with about into
than then what who whom whose which when where why how do does did done can could would should
will shall may might must have has had i you he she it we they me him her us them my your his its
our their this that these those and or not no yes there here so if as tell please know mnemo user
many much some any all most more less very just also only ever""".split())


_DET = {"a", "an", "the", "some", "any", "this", "that", "these", "those", "my", "your",
        "his", "her", "its", "our", "their", "of", "many", "two", "three"}


def _entities(mind, question):
    """The things a question is about, best first: the names it uses, the
    slots of Mnemo's own parse of it, then nouns ('a thermostat', 'koalas')."""
    kb = mind.kb
    found = sorted(mind._caps, key=len, reverse=True)          # 'Mona Lisa', 'France'
    for it in getattr(mind, "_last_parse", ()):
        if isinstance(it, L.Query):
            found += [x for x in (it.s, it.o, it.extra) if isinstance(x, str) and x]
    words = [re.sub(r"'s?$", "", w) for w in re.findall(r"[a-z][a-z'-]*", question.lower())]
    for n in (3, 2, 1):                                         # 'chemical symbol', 'gold'
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            if words[i] in _STOP or words[i + n - 1] in _STOP:
                continue
            head = L.noun(phrase)
            plural = head != phrase
            after_det = i > 0 and words[i - 1] in _DET
            if n == 1 and not (plural or after_det):
                continue                                         # 'work' in 'how does it work'
            if head in kb.senses or head in kb.aliases or head in kb.by_s:
                found.append(head)
    return list(dict.fromkeys(found))


def grounding(mind, question, limit=12):
    """What Mnemo holds about the things a question is about, as short
    lines: a definition and the most confident facts of each; another sense
    only if the usual one has nothing."""
    kb = mind.kb
    lines, used = [], []
    for w in _entities(mind, question):
        if any(w != u and w in u for u in used):
            continue                                            # 'lisa' after 'mona lisa'
        k = kb.canon(w, proper=w in mind._caps)
        for s in [k] + kb.other_senses(w, k)[:2]:
            facts = sorted((f for f in kb.out(s) if f.source != "innate" and f.truth.c >= 0.5),
                           key=lambda f: (-f.truth.c, f.r != "isa"))
            gloss = kb.definitions.get(s, "")
            if not facts and not gloss:
                continue
            if gloss:
                lines.append(f"{kb.display.get(s) or kb.word(s)}: {gloss[:160]}")
            lines += [kb.say(f) for f in facts[:4]]
            used.append(w)
            break
        if len(lines) >= limit:
            break
    return list(dict.fromkeys(lines))[:limit]


def prompt(question, facts):
    if not facts:
        return question
    return ("Facts from Mnemo's memory (they may help, or may not be relevant):\n" +
            "\n".join("- " + f for f in facts) + f"\n\nQuestion: {question}")


# ---------------------------------------------------------------- what to keep

_CHATTER = re.compile(r"^(?:i|we|you|here|sure|these|this|note|okay|ok|certainly|hope)\b|"
                      r"\b(?:i|me|my|you|your|we|our)\b", re.I)


def readable(lines, max_words=10):
    """The simple statements Mnemo's reader understands, cleaned, without
    repeats; chatter, headings and long sentences are dropped."""
    seen, out = set(), []
    for raw in lines:
        line = re.sub(r"\*\*|__", "", raw)                   # markdown emphasis
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line or line.startswith("#") or line.endswith(":") or "," in line or \
                len(line.split()) > max_words or _CHATTER.search(line):
            continue
        if not line.endswith("."):
            line = line.rstrip("!") + "."
        items = L.parse_sentence(line).items
        facts = [i for i in items if isinstance(i, L.Fact)]
        if facts and len(facts) + sum(isinstance(i, L.Event) and i.kind == "be_at"
                                      for i in items) == len(items) and \
                all(len(i.s.split()) <= 3 and len(str(i.o).split()) <= 3 for i in facts):
            key = line.lower()
            if key not in seen:
                seen.add(key)
                out.append(line)
    return out


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
