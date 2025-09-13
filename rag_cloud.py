import os
import requests
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/rag_cloud/.env"))
API_KEY = os.getenv("GEMINI_API_KEY")

BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "models/gemini-2.0-flash"

def rag_ask(query, docs):
    context = "\n".join(docs)
    prompt = f"Pytanie: {query}\n\nKONTEKST:\n{context}"

    url = f"{BASE}/{MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": API_KEY
    }
    payload = {
        "contents": [
            {
                "parts": [
                    { "text": prompt }
                ]
            }
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

if __name__ == "__main__":
    docs = [
        "RAG = Retrieval Augmented Generation, technika łączenia LLM z bazą wiedzy.",
        "Gemini to model od Google dostępny przez API."
    ]
    ans = rag_ask("Co to jest RAG i po co się go używa?", docs)
    print(ans)
