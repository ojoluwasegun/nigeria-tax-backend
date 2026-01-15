from flask import Flask, request, jsonify
from pypdf import PdfReader
import faiss
import numpy as np
import os
import requests

app = Flask(__name__)

# --------------------------------
# Hugging Face API
# --------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")

EMBEDDING_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
GENERATION_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def embed_text(texts):
    """Get embeddings from HF API"""
    response = requests.post(
        EMBEDDING_URL,
        headers=HEADERS,
        json={"inputs": texts}
    )
    response.raise_for_status()
    return response.json()

def generate_answer(prompt):
    """Generate answer using HF API"""
    response = requests.post(
        GENERATION_URL,
        headers=HEADERS,
        json={"inputs": prompt}
    )
    response.raise_for_status()
    return response.json()[0]["generated_text"]

# --------------------------------
# Load PDF
# --------------------------------
reader = PdfReader("data/data.pdf")
text = " ".join(page.extract_text() or "" for page in reader.pages)
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# --------------------------------
# Build FAISS index (once at startup)
# --------------------------------
chunk_embeddings = embed_text(chunks)
dimension = len(chunk_embeddings[0])

index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings).astype("float32"))

# --------------------------------
# API endpoint
# --------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Embed question
    q_embedding = embed_text([question])[0]

    # Search FAISS
    _, ids = index.search(
        np.array([q_embedding]).astype("float32"),
        2
    )
    context = " ".join([chunks[i] for i in ids[0]])

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    answer = generate_answer(prompt)
    return jsonify({"answer": answer})

# --------------------------------
# Local run only
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)
