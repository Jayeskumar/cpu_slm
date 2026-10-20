"""Pairing the mind with a language model: the plumbing, tested against a
stand-in server on localhost that speaks Ollama's and the OpenAI chat APIs.
The stand-in only replays canned replies, so these tests say nothing about
how well a real model answers; tools/llm_eval.py measures that."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import conftest  # noqa: F401
from mnemo.llm import LLM, readable
from mnemo.mind import Mind

REPLIES = [  # (words in the question, canned reply)
    ("mona lisa", "Leonardo da Vinci painted the Mona Lisa."),
    ("favourite food", "Penguins can fly. Penguins eat fish."),
    ("", "I don't know."),
]


def _server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            question = body["messages"][-1]["content"].lower()
            text = next(r for k, r in REPLIES if k in question)
            if self.path == "/api/chat":
                out = {"message": {"role": "assistant", "content": text}, "done": True}
            else:
                out = {"choices": [{"message": {"role": "assistant", "content": text}}]}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, seen, f"http://127.0.0.1:{httpd.server_address[1]}"


def _mind(url):
    m = Mind(learn_language=False)
    for s in ("Penguins are birds.", "Birds can fly.", "Penguins cannot fly.",
              "Penguins live in Antarctica."):
        m.hear(s)
    m.attach_llm(LLM(f"openai:tiny@{url}/v1"))
    return m


def test_model_specs_and_both_apis():
    m = LLM("ollama:gemma:2b")
    assert (m.backend, m.model, m.endpoint) == ("ollama", "gemma:2b", "http://localhost:11434")
    m = LLM("openai:qwen2.5-0.5b-instruct@http://127.0.0.1:8080/v1/")
    assert (m.backend, m.model, m.endpoint) == \
        ("openai", "qwen2.5-0.5b-instruct", "http://127.0.0.1:8080/v1")
    httpd, seen, url = _server()
    try:
        assert LLM(f"ollama:tiny@{url}").ask("Who painted the Mona Lisa?").startswith("Leonardo")
        assert LLM(f"openai:tiny@{url}/v1").ask("Who painted the Mona Lisa?").startswith(
            "Leonardo")
        assert [path for path, _ in seen] == ["/api/chat", "/v1/chat/completions"]
        for _, body in seen:                    # deterministic, with Mnemo's instructions
            assert body["messages"][0]["role"] == "system"
            assert body.get("temperature", body.get("options", {}).get("temperature")) == 0
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_memory_first_then_the_model_then_memory_again():
    httpd, seen, url = _server()
    try:
        m = _mind(url)
        assert m.hear("Can a penguin fly?").startswith("No")       # memory: the model is not asked
        assert seen == [] and m.last_source == "memory"
        assert m.hear("Who painted the Mona Lisa?") == \
            "Leonardo da Vinci painted the Mona Lisa. [tiny]"
        assert m.last_source == "tiny" and len(seen) == 1
        assert "came from the language model tiny" in m.hear("Why?")
        # what the model said, and Mnemo could read, is now in memory, marked as such
        assert m.hear("Who painted the Mona Lisa?") == \
            "Leonardo da Vinci. (I learned that from a language model.)"
        assert len(seen) == 1 and m.last_source == "memory"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_model_is_grounded_and_checked():
    httpd, seen, url = _server()
    try:
        m = _mind(url)
        reply = m.hear("What is a penguin's favourite food?")
        prompt = seen[-1][1]["messages"][-1]["content"]
        assert "Facts from Mnemo's memory" in prompt and "penguins cannot fly" in prompt
        # 'Penguins can fly' contradicts memory: pointed out, not kept; the rest is kept, weakly
        assert reply.endswith("[tiny] (I doubt part of that: penguins cannot fly.)"), reply
        assert m.kb.get("penguin", "can", "fly").truth.f < 0.5
        eat = m.kb.get("penguin", "eat", "fish")
        assert eat.source == "llm" and eat.truth.c == 0.6
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_no_model_running_is_not_an_error():
    with socket.socket() as s:                 # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    m = Mind(learn_language=False)
    m.attach_llm(LLM(f"ollama:tiny@http://127.0.0.1:{port}", timeout=5))
    reply = m.hear("Who painted the Mona Lisa?")
    assert reply.startswith("I don't know") and "did not answer" in reply
    assert m.hear("Tweety is a bird.") == "OK."      # statements never go to the model


def test_only_readable_statements_are_kept():
    assert readable(["Sure! Here are some facts:", "1. Owls are birds.",
                     "Owls hunt at night, using their excellent hearing.", "**Owls can fly.**"]) \
        == ["Owls are birds.", "Owls can fly."]
