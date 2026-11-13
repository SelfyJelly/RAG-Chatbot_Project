import os
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

# Verbindung zur Datenbank
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Muss mit dem Modell in main.py übereinstimmen!
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Ordner mit den Vorlesungsfolien
PDF_FOLDER = Path(__file__).resolve().parent / "documents"

def read_pdf(pdf_path):
    """Liest alle Seiten einer PDF-Datei aus."""

    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if not text:
                print(f"  Seite {page_number}: Kein Text gefunden")
                continue

            pages.append({
                "content": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number
                }
            })

    return pages


def chunk_content(text, chunk_size=1000, overlap=150):
    """Teilt lange Texte in kleinere Abschnitte."""

    if overlap >= chunk_size:
        raise ValueError("overlap muss kleiner als chunk_size sein")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def ingest():
    """Liest PDFs ein und speichert Embeddings in Supabase."""

    if not PDF_FOLDER.exists():
        print("PDF-Ordner nicht gefunden!")
        return

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("Keine PDF-Dateien gefunden!")
        return

    print(f"{len(pdf_files)} PDF-Dateien gefunden.")

    for pdf_path in pdf_files:
        print(f"\nVerarbeite: {pdf_path.name}")

        pages = read_pdf(pdf_path)

        for page in pages:
            chunks = chunk_content(page["content"])

            for chunk_number, chunk in enumerate(chunks, start=1):

                # Text in Vektor umwandeln
                embedding = model.encode(chunk).tolist()

                metadata = {
                    **page["metadata"],
                    "chunk": chunk_number
                }

                # In Supabase speichern
                record = {
                    "content": chunk,
                    "metadata": metadata,
                    "embedding": embedding
                }

                supabase.table("documents").insert(record).execute()

            print(
                f"  Seite {page['metadata']['page']}: "
                f"{len(chunks)} Chunks gespeichert"
            )

    print("\nAlle PDFs wurden verarbeitet!")


if __name__ == "__main__":
    ingest()