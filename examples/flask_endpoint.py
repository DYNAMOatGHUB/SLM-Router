"""Optional Flask integration; Flask is not a TinyAgent dependency.

Run: pip install flask && flask --app examples.flask_endpoint run
"""

from flask import Flask, jsonify, request

from tinyagent import TinyAgent
from tinyagent.backends import OllamaBackend
from tinyagent.demo_tools import build_registry

MAX_QUERY_CHARS = 2000

app = Flask(__name__)
# One shared agent is safe: run() keeps no state between calls (R4-17).
agent = TinyAgent(OllamaBackend("llama3:8b"), build_registry(), require_tool=True)


@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "body must be JSON with a non-empty 'query' string"}), 400
    if len(query) > MAX_QUERY_CHARS:
        return jsonify({"error": f"query longer than {MAX_QUERY_CHARS} characters"}), 413
    result = agent.run(query)
    return jsonify({
        "status": result.status,
        "text": result.text,
        "provenance": result.provenance,
        "model_calls": result.model_calls,
        "error_code": result.error_code,
    })
