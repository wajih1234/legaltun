

from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline import rag_pipeline

app = Flask(__name__)
CORS(app)


@app.route("/api/naive/query", methods=["POST"])
def query():
    """
    Endpoint principal du RAG classique
    Body: { "question": "..." }
    """
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "question is required"}), 400

    result = rag_pipeline(question)

    return jsonify({
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"],
        "nb_chunks": result["nb_chunks"],
        "chunks": result["chunks"]
    })


@app.route("/api/naive/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "RAG Classique"})


if __name__ == "__main__":
    app.run(debug=True, port=5002)  # port 5002 pour ne pas conflit avec Graph RAG (5001)