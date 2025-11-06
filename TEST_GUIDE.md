# Founders AI API Test Guide

A practical, step-by-step manual for validating the Founders AI (RAG + Adaptive + Multi-Hop reasoning) API. This guide consolidates real request/response models from the codebase (`app/models/requests.py`) and patterns described in `README.md`.

---
## 1. Pre‑Test Environment Setup

### 1.1 Required Services
- FastAPI API (`api` service)
- PostgreSQL (`postgres` service) – pgvector extension expected
- Redis (`redis` service)

### 1.2 Environment Variables
Either set `DEBUG=true` (allows missing `OPENAI_API_KEY` & `DATABASE_URL` fallback to SQLite) or provide real values.

```
OPENAI_API_KEY=sk-...        # Required if DEBUG=false
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/founders_ai
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
DEBUG=true
```

### 1.3 Start Stack (Docker)
```
docker compose up -d --build
docker compose ps
```

### 1.4 Health Verification
```
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/ | jq
```
Expected fields: `status`, `services`, `endpoints`.

---
## 2. Endpoint Inventory
Extracted from `app/api/endpoints.py`:

| Endpoint | Method | Purpose | Auth | Rate Limit | Notes |
|----------|--------|---------|------|-----------|-------|
| `/` | GET | Service info | None | No | Lists declared endpoints (includes `multi-hop` label though no explicit public route) |
| `/health` | GET | System health snapshot | None | No | Returns component status or error |
| `/query` | POST | Primary RAG + optional multi-hop fallback | None | Yes | Can trigger multi-hop chain internally |
| `/adaptive-query` | POST | Intent-driven adaptive orchestration | None | Yes | Returns enriched metadata (intent, confidence) |
| `/ingest-pdfs` | POST | Upload & ingest PDFs | None | No | Multipart form-data; stores documents | 
| `/feedback` | POST | Submit qualitative or rating feedback | None | No | Creates record; may recalc metrics |

> Missing: A direct `/multi-hop` route listed in service info but not implemented. Multi-hop is internal via `/query` when `enable_multi_hop_reasoning` or `force_multi_hop` is true. Documented below as “implicit flow.”

---
## 3. Data Models (from `app/models/requests.py`)

### 3.1 FounderQueryRequest
```
{
  "query": "string (1..10000 chars)",
  "top_k": 1..50 (default 5),
  "algorithm": "hybrid" | "vector_only" | "keyword_only",
  "similarity_threshold": 0.0..1.0 (default 0.3),
  "enable_multi_hop_reasoning": true,
  "force_multi_hop": false,
  "response_length": "short" | "normal" | "detailed",
  "use_agent": false,
  "session_id": "string or null",
  "text_only": false,
  "intent": "string or null"
}
```
Validation: `query` trimmed; cannot be blank.

### 3.2 FounderQueryResponse
Key fields: `response`, `query`, `context` (list of sources), `metadata` (intent, algorithm, timing), `source`, `response_time_ms`.

### 3.3 MultiHopReasoningResponse (internal result when multi-hop triggered)
Contains `reasoning_steps[]` with enriched trace.

### 3.4 PDFIngestionResponse
```
{
  "message": "status message",
  "document_ids": ["id1", "id2"],
  "status": "success" | "failure"
}
```

### 3.5 Feedback Submission (UserFeedback)
```
{
  "query": "original query",
  "response": "model response text",
  "intent_classified": "detected intent or empty",
  "feedback_type": "rating" | "intent_correction" | "response_quality" | "length_appropriateness" | "citation_accuracy",
  "rating": 1..5 (optional for rating-quality types),
  "correction": "string (optional)",
  "comments": "string (optional)",
  "user_id": "string (optional)",
  "session_id": "string (optional)"
}
```
Returns: `{ "message": "Feedback submitted", "feedback_id": "uuid" }` or 500 error.

---
## 4. Manual Testing Scenarios

### 4.1 Basic Service Info & Health
```
curl -s http://localhost:8000/ | jq
curl -s http://localhost:8000/health | jq
```
Expected: `status=healthy` and non-empty `services` map.

### 4.2 Simple RAG Query (No Multi-Hop)
```
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize recent startup valuation trends",
    "top_k": 3,
    "enable_multi_hop_reasoning": false,
    "force_multi_hop": false,
    "algorithm": "hybrid"
  }' | jq
```
Check: `source` should be `rag_engine`, `context` list ≤ top_k.

### 4.3 Forced Multi-Hop Query
```
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare growth metrics across fintech, healthtech and edtech early-stage startups and synthesize cross-sector insights",
    "top_k": 5,
    "force_multi_hop": true,
    "response_length": "detailed"
  }' | jq
```
Expected: `source` may still appear as `rag_engine` but response structure includes multi-hop style `metadata` and potentially multi-hop formatted context. (Multi-hop detailed object is transformed into a `FounderQueryResponse` in endpoints.) Note any large execution time.

### 4.4 Adaptive Query
```
curl -s -X POST http://localhost:8000/adaptive-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Identify strategic pivots for a SaaS startup facing churn",
    "top_k": 4,
    "response_length": "normal"
  }' | jq
```
Check: `metadata.intent`, `metadata.processing_time_ms`, `source=adaptive_rag`.

### 4.5 PDF Ingestion
Upload one or more PDFs (ensure file exists, e.g. docs/sample.pdf):
```
curl -s -X POST http://localhost:8000/ingest-pdfs \
  -F "files=@docs/sample.pdf" | jq
```
Expected: `status=success`, non-empty `document_ids`.
Edge: Non-PDF file should be silently skipped.

### 4.6 Feedback Submission (Rating)
```
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize valuation trends",
    "response": "Model response text here",
    "intent_classified": "analysis",
    "feedback_type": "rating",
    "rating": 5,
    "comments": "Clear and useful"
  }' | jq
```
Expected: success message + `feedback_id`.

### 4.7 Feedback Intent Correction
```
curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Best sector for sustainable agri-tech in LATAM",
    "response": "Response...",
    "intent_classified": "sector_analysis",
    "feedback_type": "intent_correction",
    "correction": "market_entry_strategy"
  }' | jq
```
Expected: success message; triggers metrics recalculation once enough feedback accumulates.

### 4.8 Rate Limiting (429)
Send >10 rapid `/query` requests from same client IP within 60s:
```
for i in $(seq 1 12); do 
  curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query":"Ping '$i'","top_k":1}'; 
  sleep 0.5; 
done
```
Expected: First 10 -> 200; subsequent -> 429 with JSON detail: "Rate limit exceeded...".

### 4.9 Validation Error (Empty Query)
```
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"   "}' | jq
```
Expect: 422 Unprocessable Entity (Pydantic validation message).

### 4.10 Large Query Boundary
Generate ~10k chars string to test max length:
```
LONG=$(python - <<'PY'
print('a'*10000)
PY)
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"'$LONG'"}' | jq
```
Expect: 200 or multi-hop trigger; if length > allowed, 422.

---
## 5. Structured Postman Collection (Fields Summary)

Recommended variables:
- `{{base_url}}` = `http://localhost:8000`
- `{{query_text}}`
- `{{openai_api_key}}` (if integrated via header later)

Example Requests:
1. GET {{base_url}}/
2. GET {{base_url}}/health
3. POST {{base_url}}/query (raw JSON – FounderQueryRequest)
4. POST {{base_url}}/adaptive-query
5. POST {{base_url}}/ingest-pdfs (form-data, key: files, type: File)
6. POST {{base_url}}/feedback

Include Test Scripts:
```js
// Example Postman test for /query
pm.test("Status is 200", function() { pm.response.to.have.status(200); });
pm.test("Has response field", function() {
  const json = pm.response.json();
  pm.expect(json).to.have.property("response");
  pm.expect(json).to.have.property("query");
  pm.expect(json).to.have.property("context");
});
```

---
## 6. Expected Error Modes
| Scenario | Status | Cause | Mitigation |
|----------|--------|-------|------------|
| Missing `OPENAI_API_KEY` (DEBUG=false) | 500 at startup | Config validation in `config.py` | Set key or enable DEBUG |
| Missing `DATABASE_URL` (DEBUG=false) | 500 at startup | Validator requires URL | Provide Postgres URL |
| Rate limit exceeded | 429 | >10 req/min/IP | Backoff, use different IP, or wait window reset |
| Invalid enum value | 422 | Pydantic validation failure | Use allowed enum values |
| Empty query | 422 | Field validator strips whitespace | Provide non-empty content |
| PDF upload without any valid PDFs | 200 | Non-PDFs skipped; may return 0 docs | Ensure .pdf extension |

---
## 7. Advanced / Multi-Hop Inspection
Multi-hop reasoning is internal. To confirm activation:
1. Use complex query or `force_multi_hop=true`.
2. Inspect `metadata` for complexity-related keys (e.g., `complexity_analysis`).
3. (Potential enhancement) Expose debug endpoint to return raw `MultiHopReasoningResponse` for QA.

Suggested Enhancement (Not Implemented): Add `GET /multi-hop/{chain_id}` using `ReasoningChainRequest` model to retrieve trace.

---
## 8. Performance Spot Checks
Use `time` or measure `response_time_ms` field.
```
for i in $(seq 1 3); do 
  curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query":"Growth strategy for B2B SaaS pricing", "top_k":3}' | jq '.response_time_ms';
 done
```
Record average and watch for large deviations (>3x baseline).

---
## 9. Test Coverage Checklist
- [x] Service info /health
- [x] Basic RAG query
- [x] Forced multi-hop path
- [x] Adaptive query
- [x] PDF ingestion
- [x] Feedback (rating + intent correction)
- [x] Rate limiting
- [x] Validation (empty query)
- [x] Boundary length
- [ ] (Optional) Startup metrics / DB persistence verification (requires querying DB directly)

---
## 10. Database Verification (Optional)
After PDF ingestion or queries you may inspect Postgres:
```
docker exec -it foundersai_postgres psql -U postgres -d founders_ai -c "\dt"
docker exec -it foundersai_postgres psql -U postgres -d founders_ai -c "SELECT COUNT(*) FROM startup_case_studies;"
```
Feedback table check:
```
docker exec -it foundersai_postgres psql -U postgres -d founders_ai -c "SELECT COUNT(*) FROM user_feedback;"
```

---
## 11. Troubleshooting Quick Reference
| Symptom | Action |
|---------|--------|
| `ImportError: attempted relative import` in container | Ensure Dockerfile uses `CMD ["uvicorn", "app.main:app", ...]` and volume path preserves package (`./app:/app/app`). |
| `Rate limit exceeded` frequently | Increase `rate_limit_requests` via env or space requests apart. |
| Long startup (Torch + CUDA libs) | Consider removing GPU-heavy dependencies if not needed for test; use CPU-only minimal embedding approach. |
| Empty `context` list | Verify documents ingested; run PDF ingestion before querying; check DB count. |

---
## 12. Next Improvements (QA Perspective)
- Implement explicit `/multi-hop` endpoint returning full reasoning chain.
- Add `/metrics/feedback` to expose aggregated `FeedbackMetrics`.
- Include OpenAPI examples for each model in `main.py` via `schema_extra`.
- Provide TestContainers-based integration tests (Python `pytest`) for CI.

---
## 13. Quick Curl Summary
```
# Info
curl http://localhost:8000/ | jq
curl http://localhost:8000/health | jq

# Query
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"Sample","top_k":2}' | jq

# Adaptive
curl -X POST http://localhost:8000/adaptive-query -H "Content-Type: application/json" -d '{"query":"Strategic pivot analysis"}' | jq

# Multi-hop forced
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"Cross-sector synthesis of SaaS metrics","force_multi_hop":true}' | jq

# Ingest PDF
curl -X POST http://localhost:8000/ingest-pdfs -F "files=@docs/sample.pdf" | jq

# Feedback
curl -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d '{"query":"Sample","response":"Text","feedback_type":"rating","rating":5,"intent_classified":"analysis"}' | jq
```

---
**End of Guide**
