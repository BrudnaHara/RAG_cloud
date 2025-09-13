# AI Architect Assistant

Prosty RAG (Retrieval-Augmented Generation) uruchomiony w chmurze na **Google Cloud Run**, z wykorzystaniem **Gemini API** jako modelu językowego.

Projekt został przygotowany jako pokazowe narzędzie w ramach kursu / zaliczenia certyfikacyjnego.

---

## Funkcjonalności

- **Interfejs webowy (FastAPI)**:
  - formularz do zadawania pytań,
  - możliwość dodawania dokumentów TXT/PDF lub bloków tekstu,
  - prosta baza wiedzy trzymana w pliku `store.json`,
  - historia pytań i odpowiedzi w ramach jednej sesji (przechowywana w pamięci RAM).

- **Integracja z Gemini API**:
  - model: `gemini-2.0-flash`,
  - automatyczne oczyszczanie odpowiedzi z formatowania Markdown (gwiazdki, podkreślenia itp.),
  - prompt dostosowany do stylu: *AI Architect Assistant* (technicznie, po polsku, z oznaczaniem hipotez).

- **Hosting**:
  - wdrożenie w Google Cloud Run (kontener zbudowany z Dockerfile),
  - publiczny dostęp pod wygenerowanym adresem URL,
  - konfiguracja klucza API jako zmiennej środowiskowej (`GEMINI_API_KEY`).

---

## Struktura repozytorium

rag_cloud/
├── rag_cloud.py # prosty skrypt testowy (RAG w trybie konsoli)
├── web.py # aplikacja FastAPI z interfejsem webowym
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .gitignore


- **`web.py`** – główny plik aplikacji webowej (frontend + backend).
- **`rag_cloud.py`** – minimalny RAG w trybie konsoli (prototyp).
- **`store.json`** – lokalny magazyn dokumentów (tworzony dynamicznie).
- **`requirements.txt`** – zależności Pythona.

---

## Jak uruchomić lokalnie

1. Klonuj repozytorium:
   ```bash
   git clone https://github.com/BrudnaHara/RAG_cloud.git
   cd RAG_cloud
   
2.Utwórz i aktywuj środowisko wirtualne:
python3 -m venv venv
source venv/bin/activate

3.Zainstaluj zależności:
pip install -r requirements.txt

4.Utwórz plik .env i dodaj swój klucz Gemini:
GEMINI_API_KEY=tu_wklej_swoj_klucz

5.Uruchom aplikację:
uvicorn web:app --host 127.0.0.1 --port 8080

Aplikacja będzie dostępna pod adresem: http://127.0.0.1:8080

Deploy w Google Cloud Run

1.Zbuduj i wyślij kontener:
gcloud run deploy ai-architect-assistant \
  --source . \
  --region europe-central2 \
  --platform managed \
  --allow-unauthenticated

2.Ustaw zmienną środowiskową:
gcloud run services update ai-architect-assistant \
  --region europe-central2 \
  --update-env-vars GEMINI_API_KEY="twoj_klucz"

Status

Projekt w wersji pokazowej, zrealizowany dla potrzeb zaliczenia kursu.
W planach: rozbudowa o trwałe sesje, użytkowników i lepsze zarządzanie bazą wiedzy.
