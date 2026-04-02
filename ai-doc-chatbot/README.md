# 🤖 AI Document Chatbot (RAG)

Ask questions about any PDF — answers come **only** from your document.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env → add GROQ_API_KEY (or OPENAI/GEMINI)

# 3. Run
python app.py

# 4. Open browser
# http://localhost:5000
```

## API Reference

### Upload PDF
```
POST /upload
Content-Type: multipart/form-data
Body: file=<your.pdf>
```

### Ask a Question
```
POST /chat
Content-Type: application/json
{
  "query": "What is the main conclusion?",
  "session_id": "optional-uuid-for-history"
}
```

### Get Chat History
```
GET /history?session_id=<uuid>
```

### Clear History
```
DELETE /history?session_id=<uuid>
```

### Check Status
```
GET /status
```

## Postman Sample

Import this request:
- Method: POST
- URL: http://localhost:5000/chat
- Body → raw → JSON:
```json
{
  "query": "Summarize the key points of this document",
  "session_id": "test-session-1"
}
```

## Architecture
```
PDF Upload
  └─ PDFLoader         → extract raw text
  └─ TextChunker       → split into 500-word overlapping chunks
  └─ EmbeddingService  → all-MiniLM-L6-v2 (local, no API key)
  └─ VectorStore       → FAISS index persisted to disk

User Query
  └─ EmbeddingService  → embed the question
  └─ VectorStore       → top-4 similar chunks
  └─ LLMService        → context + query → Groq/OpenAI/Gemini
  └─ ChatHistoryDB     → SQLite persistence
```
