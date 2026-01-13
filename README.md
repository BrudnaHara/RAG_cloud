# AI Architect Assistant

Asystent do streszczania i syntezy wiedzy z własnych materiałów (CAG-style / long-context). Zamiast klasycznego RAG z retrieval per zapytanie, aplikacja przekazuje treść dokumentów jako kontekst do prompta i generuje odpowiedzi na tej podstawie.

---

## Funkcjonalności

#### Interfejs webowy (FastAPI)

- Formularz do zadawania pytań
- Dodawanie dokumentów TXT lub bloków tekstu
- Prosta baza wiedzy w pliku `store.json` (przechowywana jako plik i synchronizowana z Hugging Face Hub jako dataset)
- Historia pytań i odpowiedzi w ramach jednej sesji (RAM)

#### Integracja z Gemini API

- Generowanie przez `models.generateContent` (v1beta)
- Model: `models/gemini-3-flash-preview`

- automatyczne oczyszczanie odpowiedzi z formatowania Markdown (gwiazdki, podkreślenia itp.),
- prompt dostosowany do stylu: *AI Architect Assistant* (technicznie, po polsku, z oznaczaniem hipotez).

---

## Struktura repozytorium

- **`web.py`** – główny plik aplikacji webowej (frontend + backend).
- **`store.json`** – lokalny magazyn dokumentów (tworzony dynamicznie).
- **`requirements.txt`** – zależności Pythona.

---

 
Projekt w wersji pokazowej, zrealizowany dla potrzeb zaliczenia kursu.
W planach: rozbudowa o trwałe sesje, użytkowników i lepsze zarządzanie bazą wiedzy.
