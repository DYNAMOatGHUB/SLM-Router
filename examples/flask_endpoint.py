"""Optional Flask integration; Flask is not a TinyAgent dependency."""

from flask import Flask, jsonify, request

from examples.demo_tools import build_registry
from tinyagent import TinyAgent
from tinyagent.backends import OllamaBackend

app = Flask(__name__)
agent = TinyAgent(OllamaBackend("llama3:8b"), build_registry())


@app.post("/ask")
def ask():
    result = agent.run(request.json["query"])
    return jsonify({
        "status": result.status,
        "text": result.text,
        "provenance": result.provenance,
        "model_calls": result.model_calls,
        "error_code": result.error_code,
    })
