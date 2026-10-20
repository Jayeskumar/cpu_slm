"""The local chat page: page, question and stats endpoints."""
import json
import threading
import urllib.request

import conftest  # noqa: F401
from mnemo.mind import Mind
from mnemo.serve import make_server


def _post(url, obj):
    req = urllib.request.Request(url, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def test_chat_page_answers_over_http():
    httpd = make_server(Mind(learn_language=False), port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # local, no proxy
    urllib.request.install_opener(opener)
    try:
        page = urllib.request.urlopen(base + "/", timeout=10).read().decode()
        assert "<title>Mnemo</title>" in page and "http" not in page.split("<script>")[1]
        assert _post(base + "/api/hear", {"text": "Tweety is a bird."})["reply"] == "OK."
        _post(base + "/api/hear", {"text": "Birds can fly."})
        a = _post(base + "/api/hear", {"text": "Can Tweety fly?"})
        assert a["reply"].startswith("Yes") and a["ms"] >= 0
        stats = json.load(urllib.request.urlopen(base + "/api/stats", timeout=10))
        assert stats["facts"] >= 2
    finally:
        httpd.shutdown()
        httpd.server_close()
