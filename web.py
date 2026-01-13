import os
import json
import time
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from dotenv import load_dotenv
import requests
from huggingface_hub import HfApi, hf_hub_download

# --- DODANE: wektory / FAISS ---
import numpy as np
import faiss  # IndexFlatIP + normalize_L2 => cosine similarity 

# ENV
load_dotenv(os.path.expanduser("~/rag_cloud/.env"))
API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API
BASE = "https://generativelanguage.googleapis.com/v1beta"

# --- ZMIANA: model do generowania  ---
MODEL = "models/gemini-3-flash-preview"

# --- DODANE: model do embeddingów ---
EMBED_MODEL = "models/gemini-embedding-001"  # embeddings endpoint 

# Storage: tylko lokalnie
STORE_DIR = os.path.expanduser(os.getenv("STORE_PATH", "~/rag_cloud_data"))
os.makedirs(STORE_DIR, exist_ok=True)

STORE = os.path.join(STORE_DIR, "store.json")

# --- DODANE: pliki indeksu (wariant 2) ---
EMB_FILE = os.path.join(STORE_DIR, "embeddings.npy")
INDEX_FILE = os.path.join(STORE_DIR, "faiss_index.bin")
META_FILE = os.path.join(STORE_DIR, "chunks_meta.json")  # mapowanie index->(doc_idx, chunk_idx)

# HF dataset 
HF_REPO_ID = "ChaosVariable/rag-cloud-data"
HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi()

# Limity
MAX_UPLOAD_MB = 5
HISTORY = []  # sesyjna historia pytań/odpowiedzi

# --- DODANE: parametry retrievera ---
TOP_K = 8

app = FastAPI(title="AI Architect Assistant")


def chunk(text, size=800, overlap=120):
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size - overlap)
    return [c for c in out if c.strip()]


# ===== Gemini calls =====
def embed_texts(texts, task_type: str) -> np.ndarray:
    """
    task_type: 'RETRIEVAL_DOCUMENT' albo 'RETRIEVAL_QUERY' [web:34]
    """
    texts = [t.strip() for t in texts if (t or "").strip()]
    if not texts:
        return np.zeros((0, 1), dtype="float32")

    url = f"{BASE}/{EMBED_MODEL}:batchEmbedContents"
    headers = {"Content-Type": "application/json", "X-goog-api-key": API_KEY}
    reqs = []
    for t in texts:
        reqs.append({
            "model": EMBED_MODEL,
            "content": {"parts": [{"text": t}]},
            "taskType": task_type,
        })
    payload = {"requests": reqs}

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    vecs = [e["values"] for e in data.get("embeddings", [])]
    return np.array(vecs, dtype="float32")


def rag_ask(query, docs_flat):
    context = "\n".join(docs_flat)
    prompt = (
        "Jesteś AI Architect Assistant. Odpowiadasz po polsku, technicznie i rzeczowo. "
        "Korzystaj z KONTEKSTU; jeśli czegoś nie jesteś pewien, oznacz jako HIPOTEZA. "
        "Nie używaj żadnych znaków specjalnych typu *, _, ~, ` ani formatowania Markdown. "
        f"Pytanie: {query}\n\nKONTEKST:\n{context}"
    )
    url = f"{BASE}/{MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": API_KEY}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    ans = data["candidates"][0]["content"]["parts"][0]["text"]
    ans = ans.replace("*", "").replace("_", "").replace("~", "").replace("`", "")
    return ans


# ===== HF sync =====
def _hf_pull(filename: str):
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=STORE_DIR,
        repo_type="dataset",
        token=HF_TOKEN
    )


def _hf_push(local_path: str, path_in_repo: str):
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )  # upload_file usage 


# ===== Store =====
def load_store():
    try:
        _hf_pull("store.json")
        with open(STORE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return []

    # automigracja: stare wpisy typu str → nowy obiekt
    changed, items = False, []
    for i, it in enumerate(raw if isinstance(raw, list) else []):
        if isinstance(it, dict) and "chunks" in it:
            items.append(it)
        elif isinstance(it, str):
            items.append({"name": f"legacy-{i}", "chunks": chunk(it)})
            changed = True

    if changed:
        save_store(items)
        return items

    return items or []


def save_store(items):
    data = json.dumps(items, ensure_ascii=False, indent=2)
    with open(STORE, "w", encoding="utf-8") as f:
        f.write(data)
    _hf_push(STORE, "store.json")


# ===== Index  =====
def flatten_chunks(items):
    """
    Zwraca:
      flat_texts: list[str]
      meta: list[dict]   gdzie meta[i] mapuje embedding->konkretny doc/chunk
    """
    flat_texts = []
    meta = []
    for di, doc in enumerate(items):
        for ci, ch in enumerate(doc.get("chunks", [])):
            t = ch if isinstance(ch, str) else str(ch)
            t = (t or "").strip()
            if t:
                flat_texts.append(t)
                meta.append({"doc_idx": di, "chunk_idx": ci})
    return flat_texts, meta


def rebuild_index(items):
    """
    Buduje:
      embeddings.npy (float32, znormalizowane)
      faiss_index.bin (IndexFlatIP)
      chunks_meta.json (mapowanie)
    I pushuje 3 pliki do HF dataset.
    """
    texts, meta = flatten_chunks(items)
    if not texts:
        # pusta baza: usuń lokalne pliki indeksu jeśli istnieją
        for p in (EMB_FILE, INDEX_FILE, META_FILE):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        return

    embs = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    # normalize => IP ~ cosine similarity 
    faiss.normalize_L2(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype("float32"))

    np.save(EMB_FILE, embs.astype("float32"))
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    _hf_push(EMB_FILE, "embeddings.npy")
    _hf_push(INDEX_FILE, "faiss_index.bin")
    _hf_push(META_FILE, "chunks_meta.json")


def load_index():
    """
    Próbuje pobrać index z HF. Jeśli nie ma — None.
    """
    try:
        _hf_pull("faiss_index.bin")
        _hf_pull("chunks_meta.json")
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception:
        return None, None


def retrieve_topk(query: str, items, k=TOP_K):
    index, meta = load_index()
    if index is None or not meta:
        # fallback: spróbuj odbudować i wczytać
        rebuild_index(items)
        index, meta = load_index()
        if index is None or not meta:
            return ["(brak dokumentów)"]

    q = embed_texts([query], task_type="RETRIEVAL_QUERY")
    faiss.normalize_L2(q)

    k = min(k, len(meta))
    D, I = index.search(q.astype("float32"), k)

    out = []
    for idx in I[0].tolist():
        m = meta[idx]
        di = m["doc_idx"]
        ci = m["chunk_idx"]
        # bezpieczny odczyt
        try:
            out.append(items[di]["chunks"][ci])
        except Exception:
            pass

    return out or ["(brak dokumentów)"]


def extract_txt_upload(up: UploadFile) -> str:
    name = (up.filename or "").lower()
    data = up.file.read()
    if not data:
        raise ValueError("Pusty plik lub brak danych.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Plik > {MAX_UPLOAD_MB} MB. Zmniejsz lub podziel.")
    if name.endswith(".txt") or (up.content_type or "").startswith("text/plain"):
        try:
            return data.decode("utf-8", "ignore")
        except Exception as e:
            raise ValueError(f"Nie można zdekodować TXT: {e}")
    raise ValueError("Nieobsługiwany format. Dozwolone tylko .txt")


def render(out=""):
    docs = load_store()
    li = "".join(
        f"<li>{i}. <b>{d.get('name','(bez_nazwy)')}</b> "
        f"({len(d.get('chunks',[]))} fragmentów) "
        f"<form style='display:inline' method='post' action='/del'>"
        f"<input type='hidden' name='idx' value='{i}'>"
        f"<button>Usuń cały materiał</button></form></li>"
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
  small {{ color:#666 }}
</style>
</head>
<body>
<div class="wrap">
  <h3>AI Architect Assistant</h3>
  <div class="row">
    <div class="col">
      <h4>Dodaj dokument (TYLKO TXT)</h4>
      <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file" accept=".txt,text/plain">
        <button type="submit">Dodaj do bazy</button>
      </form>
      <small>Limit {MAX_UPLOAD_MB} MB. Duże pliki podziel wcześniej.</small>
    </div>
  </div>
  <div class="row" style="margin-top:12px">
    <div class="col">
      <h4>Lub dodaj blok tekstu</h4>
      <form method="post" action="/add">
        <textarea name="doc" rows="6" placeholder="Wklej blok tekstu"></textarea><br><br>
        <button type="submit">Dodaj do bazy</button>
      </form>
      <small>Nazwa zostanie nadana automatycznie: blok-YYYYMMDD-HHMMSS.</small>
    </div>
  </div>
  <h4>Aktualna baza ({len(docs)}):</h4>
  <ol>{li if li else "<i>Brak materiałów.</i>"}</ol>
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


@app.get("/", response_class=HTMLResponse)
def index():
    return render("")


@app.get("/add")
def add_get():
    return RedirectResponse(url="/")


@app.get("/upload")
def upload_get():
    return RedirectResponse(url="/")


@app.get("/del")
def del_get():
    return RedirectResponse(url="/")


@app.get("/ask")
def ask_get():
    return RedirectResponse(url="/#ask")


@app.post("/upload", response_class=HTMLResponse)
def upload(file: UploadFile = File(...)):
    try:
        text = extract_txt_upload(file).strip()
        docs = load_store()
        docs.append({
            "name": file.filename or f"plik-{int(time.time())}.txt",
            "chunks": chunk(text)
        })
        save_store(docs)

        # --- WARIANT 2: przebuduj i wypchnij index pliki ---
        rebuild_index(docs)

        msg = f"<p>OK: dodano materiał {file.filename}.</p>"
    except ValueError as e:
        msg = f"<p>Błąd: {e}</p>"
    except Exception as e:
        msg = f"<p>Nieoczekiwany błąd: {type(e).__name__}: {e}</p>"
    return render(msg)


@app.post("/add", response_class=HTMLResponse)
def add(doc: str = Form("")):
    doc = doc.strip()
    if doc:
        docs = load_store()
        docs.append({
            "name": time.strftime("blok-%Y%m%d-%H%M%S"),
            "chunks": chunk(doc)
        })
        save_store(docs)

        # --- WARIANT 2: przebuduj i wypchnij index pliki ---
        rebuild_index(docs)

    return render("<p>Dodano blok tekstu jako osobny materiał.</p>")


@app.post("/del", response_class=HTMLResponse)
def delete(idx: int = Form(...)):
    docs = load_store()
    if 0 <= idx < len(docs):
        removed = docs.pop(idx)
        save_store(docs)

        # --- WARIANT 2: przebuduj i wypchnij index pliki ---
        rebuild_index(docs)

        return render(f"<p>Usunięto cały materiał: {removed.get('name','(bez_nazwy)')}.</p>")
    return render("<p>Indeks poza zakresem.</p>")


@app.post("/ask")
def ask(q: str = Form(...)):
    docs = load_store()

    # --- WARIANT 2:  retrieval top-k przez FAISS ---
    top_chunks = retrieve_topk(q, docs, k=TOP_K)

    ans = rag_ask(q, top_chunks)
    HISTORY.append((q, ans))
    return RedirectResponse(url="/#ask", status_code=303)


from fastapi.responses import PlainTextResponse

@app.get("/debug", response_class=PlainTextResponse)
def debug():
    return (
        f"mode=FILE path={STORE_DIR}\n"
        f"store={STORE}\n"
        f"embeddings={EMB_FILE}\n"
        f"index={INDEX_FILE}\n"
        f"meta={META_FILE}\n"
    )
