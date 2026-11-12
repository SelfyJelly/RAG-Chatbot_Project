import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sample content representing website pages
CONTENT = [
    {
        "content": "The AWS Cloud Club is a student-led community focused on cloud computing, AI/ML, and software development. We host workshops, hackathons, and study groups throughout the academic year.",
        "metadata": {"source": "landing_page", "section": "About Us"}
    },
    {
        "content": "To apply for membership, visit our Apply Now page and fill out the application form. Applications are open at the start of each semester. You need to be a currently enrolled student.",
        "metadata": {"source": "apply_now", "section": "How to Apply"}
    },
    {
        "content": "Our Cloud Engineering department builds and maintains cloud infrastructure projects. The AI/ML department focuses on machine learning research and applications. The Web Development department handles our online presence.",
        "metadata": {"source": "core_team", "section": "Departments"}
    },
    {
        "content": "Cloud 101 Workshop is our beginner-friendly introduction to AWS services. It covers EC2, S3, and Lambda basics. The next session is scheduled for the first week of each month.",
        "metadata": {"source": "events", "section": "Cloud 101"}
    },
    {
        "content": "The AI Study Group meets every Wednesday at 6 PM in the Computer Science building, Room 204. We cover topics from basic neural networks to advanced LLM applications.",
        "metadata": {"source": "events", "section": "AI Study Group"}
    },
    {
        "content": "Open positions include Cloud Engineering Lead, AI/ML Research Assistant, and Web Developer. All positions require a minimum commitment of 5 hours per week.",
        "metadata": {"source": "apply_now", "section": "Open Positions"}
    },
    {
        "content": "Our FAQ: Membership is free for all enrolled students. You do not need prior cloud experience to join. We provide all learning resources and AWS credits for projects.",
        "metadata": {"source": "faq", "section": "General FAQ"}
    },
]

def chunk_content(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest():
    """Embed and upsert all content into Supabase."""
    print("Starting ingestion...")

    for i, item in enumerate(CONTENT):
        chunks = chunk_content(item["content"])

        for j, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()

            record = {
                "content": chunk,
                "metadata": item["metadata"],
                "embedding": embedding,
            }

            supabase.table("documents").upsert(record).execute()
            print(f"  Ingested chunk {j+1} from {item['metadata']['source']}/{item['metadata']['section']}")

    print(f"\nIngestion complete! {len(CONTENT)} documents processed.")


if __name__ == "__main__":
    ingest()