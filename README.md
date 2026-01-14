# AI Architect Assistant

Asystent do ekstrakcji informacji, syntezy wiedzy i analizy materiałów (CAG-style) 
Zamiast klasycznego RAG z retrieval per zapytanie, aplikacja przekazuje treść dokumentów jako kontekst do prompta i generuje odpowiedzi na tej podstawie

---

## Funkcjonalności

#### Interfejs webowy (FastAPI)

-Formularz do zadawania pytań
-Dodawanie dokumentów TXT lub bloków tekstu
-Historia pytań i odpowiedzi w ramach jednej sesji (RAM)
-Zarządzanie materiałami (lista, liczba chunków, usuwanie całości)

#### Architektura

-Backend: FAST API
-Trwała baza wiedzy synchronizowana z Hugging Face (dataset)
-Automatyczny preprocesing i chunking treści
-Brak retrieval i embeddingów
-Brak zewnątrznego WebUI - aplikacja hostowana jako Space na Hugging Face
-Model: gemini-3-flash-preview

#### Styl odpowiedzi
-automatyczne oczyszczanie odpowiedzi z formatowania Markdown (gwiazdki, podkreślenia itp.),
-prompt dostosowany do stylu: *AI Architect Assistant* (technicznie, po polsku, z oznaczaniem hipotez).

---

## Struktura repozytorium

- **`web.py`** – główny plik aplikacji webowej (frontend + backend).
- **`store.json`** – lokalny magazyn dokumentów (tworzony dynamicznie).
- **`requirements.txt`** – zależności Pythona.

---


