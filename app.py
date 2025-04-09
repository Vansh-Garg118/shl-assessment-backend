from flask import Flask, request, jsonify
from flask_cors import CORS  # NEW
import pickle
import faiss
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load FAISS index and metadata
index = faiss.read_index("catalog_index_cosine.faiss")

with open("catalog_texts.pkl", "rb") as f:
    catalog_texts = pickle.load(f)

with open("catalog_metadata.pkl", "rb") as f:
    catalog_metadata = pickle.load(f)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Utility to extract plain text from a URL (e.g. job description page)
def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # If it's a LinkedIn job URL
        if "linkedin.com" in url:
            containers = soup.find_all(class_=["jobs-description__container", "jobs-description__container--condensed"])
            text = " ".join([container.get_text(separator=' ') for container in containers])
        else:
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=' ')

        return ' '.join(text.split())

    except Exception as e:
        print(f"[!] Error scraping URL: {e}")
        return ""

# Semantic search function
def search_catalog_cosine(user_query, top_k=10, include_title=True):
    urls = re.findall(r'(https?://\S+)', user_query)
    for url in urls:
        scraped = extract_text_from_url(url)
        if scraped:
            user_query += f" {scraped}"

    query_embedding = model.encode([user_query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_embedding, k=top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        item = catalog_metadata[idx]

        # Extract numeric value from duration
        raw_duration = item.get("Assessment Length", "")
        match = re.search(r'\d+(\.\d+)?', str(raw_duration))
        numeric_duration = float(match.group()) if match else None
        if numeric_duration and numeric_duration.is_integer():
            numeric_duration = int(numeric_duration)

        # Split test type into list
        raw_test_type = item.get("Test Type", "")
        test_type_list = [s.strip() for s in raw_test_type.split(",") if s.strip()]

        result_item = {
            "url": item.get("URL"),
            "description": item.get("Description"),
            "duration": numeric_duration,
            "remote_support": item.get("Remote Testing"),
            "adaptive_support": "Yes" if item.get("Adaptive/IRT Support") == 1 else "No",
            "test_type": test_type_list
        }

        if include_title:
            result_item["title"] = item.get("Title")

        results.append(result_item)

    return {"recommended_assessments": results}


@app.route('/', methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the homepage!"})

@app.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Please provide a 'query' field in JSON body"}), 400

    query = data["query"]
    recommendations = search_catalog_cosine(query,include_title=False)
    return jsonify(recommendations)

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Please provide a 'query' field in JSON body"}), 400

    query = data["query"]
    recommendations = search_catalog_cosine(query)
    return jsonify(recommendations)


# Run the API server locally
if __name__ == "__main__":
    app.run(port = 8080, debug=True)
