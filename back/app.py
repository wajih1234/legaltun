# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS

from schema_fetcher import get_schema
from llm_cypher import generate_cypher
from cypher_runner import run_cypher, validate_cypher_read_only
from answer_generator import generate_answer
from entity_extractor import resolve_entities


app = Flask(__name__)


CORS(app)

# @app.route defines a URL endpoint
# methods=["POST"] means this endpoint only accepts POST requests
@app.route("/api/query", methods=["POST"])
def query():

    # request.get_json() reads the JSON body the client sent
    # example body: { "question": "how many laws are there?" }
    data = request.get_json()

    # get the question from the body
    # if "question" key doesn't exist, default to empty string
    question = data.get("question", "")

    # basic validation — don't process empty questions
    if not question:
        # jsonify() converts a Python dict to a JSON response
        # 400 is the HTTP status code for "bad request"
        return jsonify({"error": "question is required"}), 400

    # step 2: resolve user entities against graph names (debug only for now)
    resolved_entities = resolve_entities(question)
    print(f"Resolved entities: {resolved_entities}")

    cypher = generate_cypher(question)

    # step 3: handle the case where Claude said it can't answer
    if cypher == "INVALID":
        return jsonify({
            "cypher": None,
            "results": [],
            "answer": "Je n'ai pas pu générer une requête pour cette question. Veuillez la reformuler.",
            "count": 0
        })

    # step 4: validate query safety before execution
    is_valid, reason = validate_cypher_read_only(cypher)
    if not is_valid:
        return jsonify({
            "cypher": cypher,
            "results": [],
            "answer": "La requête générée a été bloquée par la validation de sécurité.",
            "count": 0,
            "error": reason
        }), 400

    # step 5: run the Cypher on Neo4j
    results = run_cypher(cypher)

    # step 6: ask the model to summarize the results
    answer = generate_answer(question, cypher, results)

    # step 7: return everything to the client
    return jsonify({
        "cypher": cypher,
        "results": results,
        "answer": answer,
        "count": len(results)
    })


# health check endpoint — useful to test if the server is running
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})



if __name__ == "__main__":
    app.run(debug=True, port=5001)