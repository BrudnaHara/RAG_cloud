# AI Architect Assistant

Prosty RAG (Retrieval-Augmented Generation) z wykorzystaniem **Gemini API** jako modelu językowego.

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

---

## Struktura repozytorium

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


Projekt w wersji pokazowej, zrealizowany dla potrzeb zaliczenia kursu.
W planach: rozbudowa o trwałe sesje, użytkowników i lepsze zarządzanie bazą wiedzy.
