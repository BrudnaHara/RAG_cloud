import os, json, io
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import requests
from pypdf import PdfReader
from fastapi.responses import RedirectResponse


load_dotenv(os.path.expanduser("~/rag_cloud/.env"))
API_KEY = os.getenv("GEMINI_API_KEY")
BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "models/gemini-2.0-flash"
STORE = os.path.expanduser("~/rag_cloud/store.json")

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
    prompt = f"Pytanie: {query}\n\nKONTEKST:\n{context}"
    url = f"{BASE}/{MODEL}:generateContent"
    headers = {"Content-Type":"application/json","X-goog-api-key":API_KEY}
    payload = {"contents":[{"parts":[{"text":prompt}]}]}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

app = FastAPI(title="RAG Cloud (Gemini, persistent)")

def render(out=""):
    docs = load_store()
    li = "".join(
        f"<li>{i}. {d[:120]}{'...' if len(d)>120 else ''} "
        f"<form style='display:inline' method='post' action='/del'><input type='hidden' name='idx' value='{i}'><button>Usuń</button></form></li>"
        for i, d in enumerate(docs)
    )
    html = f"""<!doctype html><html><body style="font-family:monospace;max-width:900px;margin:24px">
<h3>RAG Cloud (Gemini) — baza trwała</h3>

<h4>Dodaj dokument (TXT/PDF)</h4>
<form method="post" action="/upload" enctype="multipart/form-data">
<input type="file" name="file" accept=".txt,.pdf">
<button type="submit">Dodaj do bazy</button>
</form>

<h4>Lub wklej blok tekstu</h4>
<form method="post" action="/add">
<textarea name="doc" rows="6" style="width:100%" placeholder="Wklej blok tekstu"></textarea><br><br>
<button type="submit">Dodaj do bazy</button>
</form>

<h4>Aktualna baza ({len(docs)}):</h4>
<ol>{li}</ol>

<h4>Zapytanie</h4>
<form method="post" action="/ask">
<input name="q" style="width:100%" placeholder="Pytanie"><br><br>
<button type="submit">Wyślij</button>
</form>

{out}
</body></html>"""
    return HTMLResponse(html)

@app.get("/", response_class=HTMLResponse)
def index(): return render("")

@app.get("/add")
def add_get():
    return RedirectResponse(url="/")

@app.get("/upload")
def upload_get():
    return RedirectResponse(url="/")

@app.get("/del")
def del_get():
    return RedirectResponse(url="/")


@app.post("/upload", response_class=HTMLResponse)
def upload(file: UploadFile = File(...)):
    text = extract_from_upload(file).strip()
    if text:
        items = load_store()
        items.extend(chunk(text))
        save_store(items)
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

@app.post("/ask", response_class=HTMLResponse)
def ask(q: str = Form(...)):
    docs = load_store() or ["(brak dokumentów)"]
    ans = rag_ask(q, docs)
    return render(f"<h4>Odpowiedź:</h4><pre>{ans}</pre>")
