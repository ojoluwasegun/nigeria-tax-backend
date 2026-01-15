from flask import Flask, request, jsonify
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import requests

app = Flask(__name__)

# -------------------------------
# Hugging Face Inference API
# -------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TOKEN = os.environ.get("HF_TOKEN")  # ONLY used here

def generate_answer(prompt):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": prompt})
    response.raise_for_status()
    return response.json()[0]["generated_text"]

# -------------------------------
# Load PDF
# -------------------------------
reader = PdfReader("data/data.pdf")
text = " ".join(page.extract_text() or "" for page in reader.pages)
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# -------------------------------
# Load embedding model (NO TOKEN)
# -------------------------------
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    use_auth_token=False
)

embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# -------------------------------
# API endpoint
# -------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    q_emb = embedder.encode([question])
    _, ids = index.search(np.array(q_emb), 2)
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

if __name__ == "__main__":
    app.run()
