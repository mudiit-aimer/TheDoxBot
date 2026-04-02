"""
app.py
------
Flask application — the main entry point.

Endpoints:
  POST /upload        → Upload + index a PDF
  POST /chat          → Ask a question about the document
  GET  /history       → Retrieve chat history for a session
  DELETE /history     → Clear chat history for a session
  GET  /status        → Check if a document is loaded
  GET  /              → Serve the HTML frontend
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from services.retrieval_service import RetrievalService
from services.llm_service import LLMService
from database.chat_history import ChatHistoryDB

# Load environment variables from .env
load_dotenv()

# ------------------------------------------------------------------ #
#  App setup                                                          #
# ------------------------------------------------------------------ #

app = Flask(__name__)
CORS(app)   # Allow cross-origin requests (needed for frontend dev)

app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_MB", 20)) * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ------------------------------------------------------------------ #
#  Service initialisation (singletons shared across requests)         #
# ------------------------------------------------------------------ #

retrieval = RetrievalService(
    chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
    top_k=int(os.getenv("TOP_K_RESULTS", 4)),
)
llm = LLMService()
db  = ChatHistoryDB()


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def error(message: str, code: int = 400):
    return jsonify({"error": message}), code

def success(data: dict, code: int = 200):
    return jsonify({"success": True, **data}), code


# ------------------------------------------------------------------ #
#  Routes                                                             #
# ------------------------------------------------------------------ #

@app.route("/")
def index():
    """Serve the HTML frontend."""
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def status():
    """Check whether a document has been loaded into the vector store."""
    return jsonify({
        "ready": retrieval.is_ready,
        "document": retrieval.current_doc_name,
        "vectors": retrieval.vector_store.doc_count,
    })


@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload a PDF and index it.

    Form data:
      file: the PDF file

    Returns:
      JSON with chunk count, page count, etc.
    """
    if "file" not in request.files:
        return error("No file provided. Send a PDF in the 'file' field.")

    file = request.files["file"]
    if file.filename == "":
        return error("Empty filename.")
    if not allowed_file(file.filename):
        return error("Only PDF files are supported.")

    # Save file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Run ingestion pipeline
    try:
        result = retrieval.ingest_pdf(file_path)
    except Exception as e:
        return error(f"Failed to process PDF: {str(e)}", 500)

    return success(result, 201)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Ask a question about the uploaded document.

    JSON body:
      query      (required): The user's question.
      session_id (optional): UUID to maintain chat history. Auto-created if absent.

    Returns:
      JSON with answer, sources, and session_id.
    """
    if not retrieval.is_ready:
        return error("No document loaded. Please upload a PDF first.", 503)

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return error("'query' field is required.")

    # Session management
    session_id = body.get("session_id") or str(uuid.uuid4())
    db.create_session(session_id, doc_name=retrieval.current_doc_name)

    # Retrieve relevant context from the document
    try:
        context, sources = retrieval.get_context_string(query)
    except Exception as e:
        return error(f"Retrieval failed: {str(e)}", 500)

    # Get previous messages for multi-turn conversation
    history = db.get_history_for_llm(session_id, last_n=6)

    # Call the LLM
    try:
        answer = llm.answer(query=query, context=context, chat_history=history)
    except Exception as e:
        return error(f"LLM call failed: {str(e)}", 500)

    # Persist messages
    db.add_message(session_id, role="user",      content=query)
    db.add_message(session_id, role="assistant", content=answer)

    return success({
        "answer":     answer,
        "sources":    sources,
        "session_id": session_id,
    })


@app.route("/history", methods=["GET"])
def get_history():
    """
    Retrieve chat history for a session.
    Query param: session_id
    """
    session_id = request.args.get("session_id", "")
    if not session_id:
        return error("session_id query param is required.")
    messages = db.get_messages(session_id)
    return jsonify({"session_id": session_id, "messages": messages})


@app.route("/history", methods=["DELETE"])
def clear_history():
    """Clear all messages for a session."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return error("session_id query param is required.")
    db.clear_session(session_id)
    return success({"message": "History cleared."})


# ------------------------------------------------------------------ #
#  Run                                                                #
# ------------------------------------------------------------------ #


    print("=" * 50)
    print(" AI Document Chatbot - RAG Pipeline")
    print("=" * 50)

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
      
