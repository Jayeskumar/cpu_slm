"""A local chat page for a mind: python3 -m mnemo serve --brain models/kbrain

Standard library only, no network access beyond the local socket, nothing
loaded from the internet.  POST /api/hear {"text": ...} -> {"reply", "ms"};
GET /api/stats; POST /api/save.  Binds to 127.0.0.1 unless told otherwise.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mnemo</title>
<style>
:root{--bg:#fafaf7;--panel:#fff;--ink:#1f2328;--muted:#6b7078;--line:#e3e3dc;
--me:#e8f0fe;--it:#f3f3ee;--accent:#2f6fde;--why:#7a5af5}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--panel:#1d2024;--ink:#e6e6e3;
--muted:#9aa0a6;--line:#2c3036;--me:#1f2f4d;--it:#24272c;--accent:#7aa7ff;--why:#b29cff}}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",
sans-serif;display:flex;flex-direction:column}
header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;gap:12px;
align-items:baseline;flex-wrap:wrap}
header b{font-size:18px}header span{color:var(--muted);font-size:13px}
#log{flex:1;overflow-y:auto;padding:18px;max-width:860px;width:100%;margin:0 auto}
.msg{margin:10px 0;display:flex}.msg.me{justify-content:flex-end}
.msg>div{max-width:min(78%,640px);min-width:0;display:flex;flex-direction:column;
align-items:flex-start}.msg.me>div{align-items:flex-end}
.bubble{padding:9px 13px;border-radius:14px;white-space:pre-wrap;overflow-wrap:break-word}
.me .bubble{background:var(--me)}.it .bubble{background:var(--it)}
.meta{font-size:12px;color:var(--muted);margin-top:3px}
.meta button{border:0;background:none;color:var(--why);cursor:pointer;padding:0;font:inherit}
#chips{display:flex;gap:8px;overflow-x:auto;padding:0 18px 10px;max-width:860px;width:100%;
margin:0 auto;scrollbar-width:thin}
#chips button{flex:none;border:1px solid var(--line);background:var(--panel);color:var(--ink);
border-radius:999px;padding:5px 11px;font-size:13px;cursor:pointer;white-space:nowrap}
form{display:flex;gap:8px;padding:12px 18px 18px;max-width:860px;width:100%;margin:0 auto}
input{flex:1;padding:11px 13px;border-radius:12px;border:1px solid var(--line);
background:var(--panel);color:var(--ink);font:inherit;min-width:0}
form button{padding:0 16px;border-radius:12px;border:0;background:var(--accent);color:#fff;
font:inherit;cursor:pointer}
</style></head><body>
<header><b>Mnemo</b><span id="stats">a tiny mind on one CPU core</span></header>
<div id="log" aria-live="polite"></div>
<div id="chips"></div>
<form id="f" autocomplete="off"><input id="q" placeholder="Teach me, tell me a story, or ask"
 aria-label="Message"><button>Send</button></form>
<script>
const ex=["Is a whale a fish?","Which animals can fly?","What kind of animal is a frog?",
"Name some planets.","Is Tokyo in Europe?","What is the difference between a bat and a bird?",
"Kiri is a kiwi. Can Kiri fly?","Mary went to the kitchen. Where is Mary?","What is 12 times 7?"];
const log=document.getElementById("log"),q=document.getElementById("q");
function add(who,text,meta){const d=document.createElement("div");d.className="msg "+who;
const b=document.createElement("div");b.className="bubble";b.textContent=text;
const w=document.createElement("div");w.appendChild(b);
if(meta){const m=document.createElement("div");m.className="meta";m.textContent=meta+" ";
if(who==="it"&&text!=="OK."&&!text.startsWith("I don't")){const y=document.createElement("button");
y.textContent="why?";y.onclick=()=>send("Why?");m.appendChild(y)}w.appendChild(m)}
d.appendChild(w);log.appendChild(d);log.scrollTop=log.scrollHeight}
async function stats(){try{const r=await fetch("api/stats");const s=await r.json();
document.getElementById("stats").textContent=s.facts.toLocaleString()+" facts about "+
s.concepts.toLocaleString()+" concepts \\u00b7 "+s.events+" events remembered"}catch(e){}}
async function send(text){if(!text.trim())return;add("me",text);
try{const r=await fetch("api/hear",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({text})});const a=await r.json();add("it",a.reply||"(no reply)",
a.ms.toFixed(1)+" ms");stats()}catch(e){add("it","(the mind is not reachable)")}}
document.getElementById("f").onsubmit=e=>{e.preventDefault();const t=q.value;q.value="";send(t)};
const chips=document.getElementById("chips");for(const t of ex){const c=document.createElement(
"button");c.type="button";c.textContent=t;c.onclick=()=>send(t);chips.appendChild(c)}
stats();q.focus();
</script></body></html>"""


def make_server(mind, host="127.0.0.1", port=8765, brain=None):
    """An HTTP server for the chat page; port 0 picks a free port."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/stats":
                with lock:
                    kb = mind.kb
                    self._json({"facts": sum(1 for f in kb.facts.values()
                                             if f.source != "innate"),
                                "concepts": len(kb.by_s), "events": len(mind.world.events)})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 100_000:
                return self._json({"error": "too long"}, 413)
            try:
                data = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json({"error": "bad json"}, 400)
            if self.path == "/api/hear":
                text = str(data.get("text", ""))[:2000]
                with lock:
                    t0 = time.perf_counter()
                    reply = mind.hear(text)
                    ms = (time.perf_counter() - t0) * 1000
                self._json({"reply": reply, "ms": ms})
            elif self.path == "/api/save" and brain:
                with lock:
                    mind.save(brain)
                self._json({"saved": brain})
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(mind, host="127.0.0.1", port=8765, brain=None):
    httpd = make_server(mind, host, port, brain)
    print(f"Mnemo is listening on http://{host}:{httpd.server_address[1]}/  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
