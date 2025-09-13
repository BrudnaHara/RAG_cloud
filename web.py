import os, json, io
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from dotenv import load_dotenv
import requests
from pypdf import PdfReader

load_dotenv(os.path.expanduser("~/rag_cloud/.env"))
API_KEY = os.getenv("GEMINI_API_KEY")
BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "models/gemini-2.0-flash"
STORE = os.path.expanduser("~/rag_cloud/store.json")

HISTORY = []  # historia w RAM (kasuje się po restarcie kontenera)

def load_store():
    if not os.path.exists(STORE): return []
    with open(STORE, "r", encoding="utf-8") as f: return json.load(f)

def save_store(items):
    with open(STORE, "w", encoding="utf-8") as f: json.dump(items, f, ensure_ascii=False, indent=2)

def chunk(text, size=800, overlap=120):
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size - overlap)
    return [c for c in out if c.strip()]

def extract_from_upload(up: UploadFile):
    name = (up.filename or "").lower()
    data = up.file.read()
    if name.endswith(".txt"):
        return data.decode("utf-8", "ignore")
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return ""
def rag_ask(query, docs):
    context = "\n".join(docs)
    prompt = f"Jesteś AI Architect Assistant. Odpowiadasz po polsku, technicznie i rzeczowo. " \
             f"Korzystaj z KONTEKSTU poniżej; jeśli czegoś nie jesteś pewien, oznacz jako HIPOTEZA. " \
             f"Nie używaj żadnych znaków specjalnych typu *, _, ~, ` ani formatowania Markdown. " \
             f"Pytanie: {query}\n\nKONTEKST:\n{context}"
    url = f"{BASE}/{MODEL}:generateContent"
    headers = {"Content-Type":"application/json","X-goog-api-key":API_KEY}
    payload = {"contents":[{"parts":[{"text":prompt}]}]}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    ans = data["candidates"][0]["content"]["parts"][0]["text"]
    ans = ans.replace("*","").replace("_","").replace("~","").replace("`","")
    return ans


app = FastAPI(title="AI Architect Assistant")

def render(out=""):
    docs = load_store()
    li = "".join(
        f"<li>{i}. {d[:120]}{'...' if len(d)>120 else ''} "
        f"<form style='display:inline' method='post' action='/del'><input type='hidden' name='idx' value='{i}'><button>Usuń</button></form></li>"
        for i, d in enumerate(docs)
    )
    hist = "".join(
        f"<div class='qa'><div class='q'><b>Pytanie:</b><br><pre>{q}</pre></div>"
        f"<div class='a'><b>Odpowiedź:</b><br><pre>{a}</pre></div></div>"
        for q, a in HISTORY
    )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AI Architect Assistant</title>
<style>
  body {{ font-family: monospace; background:#fafafa; }}
  .wrap {{ max-width: 900px; margin:24px auto; background:#fff; padding:16px 20px; border:1px solid #eee; border-radius:10px; }}
  h3 {{ margin-top:0 }}
  textarea, input[type="text"] {{ width:100%; box-sizing:border-box; }}
  pre {{ white-space:pre-wrap; word-wrap:break-word; margin:8px 0; }}
  .qa {{ border-top:1px dashed #ddd; padding-top:12px; margin-top:12px; }}
  .row {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .col {{ flex:1 1 100%; }}
  ol {{ padding-left: 20px; }}
  button {{ cursor:pointer }}
</style>
</head>
<body>
<div class="wrap">
  <h3>AI Architect Assistant</h3>

  <div class="row">
    <div class="col">
      <h4>Dodaj dokument (TXT/PDF)</h4>
      <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file" accept=".txt,.pdf">
        <button type="submit">Dodaj do bazy</button>
      </form>
    </div>
  </div>

  <div class="row" style="margin-top:12px">
    <div class="col">
      <h4>Lub dodaj blok tekstu</h4>
      <form method="post" action="/add">
        <textarea name="doc" rows="6" placeholder="Wklej blok tekstu"></textarea><br><br>
        <button type="submit">Dodaj do bazy</button>
      </form>
    </div>
  </div>

  <h4>Aktualna baza ({len(docs)}):</h4>
  <ol>{li}</ol>

  <h4>Historia (ta sesja)</h4>
  <div>{hist if hist else "<i>Brak pytań w tej sesji.</i>"}</div>

  {out}

  <h4 id="ask">Pytanie</h4>
  <form method="post" action="/ask">
    <input type="text" name="q" placeholder="Pytanie">
    <br><br><button type="submit">Wyślij</button>
  </form>
</div>
</body>
</html>"""
    return HTMLResponse(html)

@app.get("/ask")
def ask_get():
    return RedirectResponse(url="/#ask")

@app.get("/", response_class=HTMLResponse)
def index(): return render("")

@app.get("/add")
def add_get(): return RedirectResponse(url="/")

@app.get("/upload")
def upload_get(): return RedirectResponse(url="/")

@app.get("/del")
def del_get(): return RedirectResponse(url="/")

@app.post("/upload", response_class=HTMLResponse)
def upload(file: UploadFile = File(...)):
    text = extract_from_upload(file).strip()
    if text:
        items = load_store(); items.extend(chunk(text)); save_store(items)
        msg = f"<p>Dodano z pliku: {file.filename} (pofragmentowano).</p>"
    else:
        msg = "<p>Nieobsługiwany format lub pusty plik.</p>"
    return render(msg)

@app.post("/add", response_class=HTMLResponse)
def add(doc: str = Form("")):
    doc = doc.strip()
    if doc:
        items = load_store(); items.extend(chunk(doc)); save_store(items)
    return render("<p>Dodano.</p>")

@app.post("/del", response_class=HTMLResponse)
def delete(idx: int = Form(...)):
    items = load_store()
    if 0 <= idx < len(items): items.pop(idx); save_store(items)
    return render("<p>Usunięto.</p>")

@app.post("/ask")
def ask(q: str = Form(...)):
    docs = load_store() or ["(brak dokumentów)"]
    ans = rag_ask(q, docs)
    HISTORY.append((q, ans))
    return RedirectResponse(url="/#ask", status_code=303)
