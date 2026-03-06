# Semantic PDF RAG CLI

## Enterprise-Grade Retrieval-Augmented Generation System

**Author:** Danielle Oliveira\
**Date:** 2026-03-03\
**Language:** Python\
**Database:** PostgreSQL + pgVector\
**Framework:** LangChain\
**Execution Environment:** Docker & Docker Compose

------------------------------------------------------------------------

# 1. Executive Summary

This project implements a production-grade Retrieval-Augmented
Generation (RAG) system capable of:

-   Ingesting a PDF document
-   Splitting content into semantically coherent chunks
-   Generating embeddings
-   Persisting vectors into PostgreSQL (pgVector)
-   Enabling semantic search
-   Responding to CLI-based questions strictly grounded in document
    context

The system is designed to:

-   Prevent hallucinations
-   Enforce context-bounded reasoning
-   Remain provider-agnostic (OpenAI / Gemini)
-   Optimize token usage
-   Maintain architectural clarity

------------------------------------------------------------------------

# 2. Problem Statement

Large Language Models (LLMs):

-   Do not inherently know private documents
-   May hallucinate when lacking information
-   Cannot guarantee grounded responses without retrieval mechanisms

This system mitigates these issues through:

-   Controlled semantic retrieval
-   Strict prompt constraints
-   Fail-safe response rules
-   Context-only response enforcement

------------------------------------------------------------------------

# 3. System Architecture

## 3.1 High-Level Architecture

User CLI\
↓\
Chat Layer (chat.py)\
↓\
Search Orchestrator (search.py)\
↓\
Vector Store (PostgreSQL + pgVector)\
↑\
Ingestion Pipeline (ingest.py)

------------------------------------------------------------------------

## 3.2 Ingestion Flow

1.  Load PDF via PyPDFLoader
2.  Split document:
    -   Chunk size: 1000 characters
    -   Overlap: 150 characters
3.  Generate embeddings
4.  Persist vectors using PGVector
5.  Store metadata and content

------------------------------------------------------------------------

## 3.3 Search Flow

1.  Receive user question via CLI
2.  Vectorize question
3.  Perform similarity search (k=10)
4.  Concatenate retrieved context
5.  Construct controlled prompt
6.  Invoke LLM
7.  Return grounded response

------------------------------------------------------------------------

# 4. Prompt Engineering Strategy

## 4.1 Prompt Template

CONTEXTO: {retrieved_context}

REGRAS: - Responda somente com base no CONTEXTO. - Se a informação não
estiver explicitamente no CONTEXTO, responda: "Não tenho informações
necessárias para responder sua pergunta." - Nunca invente ou use
conhecimento externo. - Nunca produza opiniões ou interpretações além do
que está escrito.

PERGUNTA DO USUÁRIO: {user_question}

------------------------------------------------------------------------

## 4.2 Anti-Hallucination Controls

-   Context anchoring
-   Explicit failure response
-   Zero external knowledge
-   Deterministic constraint-first prompting

------------------------------------------------------------------------

# 5. Token Optimization Strategy

  Strategy             Justification
  -------------------- ---------------------------------
  Chunk size = 1000    Balance granularity and context
  Overlap = 150        Preserve semantic continuity
  Top-K = 10           Control prompt size
  No chat history      Avoid context explosion
  Lightweight models   Reduce cost and latency

------------------------------------------------------------------------

# 6. Database Design

``` sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536),
    metadata JSONB
);
```

Embedding dimensions depend on provider:

-   OpenAI: 1536
-   Gemini: 3072

------------------------------------------------------------------------

# 7. Provider Configuration

## OpenAI

-   Embeddings: text-embedding-3-small
-   LLM: gpt-4o-mini

## Gemini

-   Embeddings: models/embedding-001
-   LLM: gemini-2.5-flash-lite

Provider selection via environment variable.

------------------------------------------------------------------------

# 8. Project Structure

    ├── docker-compose.yml
    ├── requirements.txt
    ├── .env
    ├── src/
    │   ├── ingest.py
    │   ├── search.py
    │   ├── chat.py
    ├── document.pdf
    └── README.md

------------------------------------------------------------------------

# 9. Execution Guide

## Step 1 --- Start Database

docker compose up -d

## Step 2 --- Ingest PDF

python src/ingest.py

## Step 3 --- Run CLI Chat

python src/chat.py

------------------------------------------------------------------------

# 10. Non-Functional Requirements

-   Deterministic behavior
-   Reproducibility
-   Provider abstraction
-   Token efficiency
-   Context isolation
-   Docker-based portability

------------------------------------------------------------------------

# 11. Architectural Decisions (ADR Summary)

ADR-001: Use RAG instead of fine-tuning\
ADR-002: Provider-agnostic embedding layer\
ADR-003: CLI-first interaction\
ADR-004: Strict prompt control\
ADR-005: Token-efficient retrieval strategy

------------------------------------------------------------------------

# 12. Limitations

-   Single-document ingestion
-   No re-ranking
-   No dynamic threshold
-   No response confidence scoring

Future improvements may include:

-   Dynamic Top-K
-   Context compression
-   Re-ranking models
-   Multi-document support
-   Evaluation metrics layer

------------------------------------------------------------------------

# 13. Conclusion

This project demonstrates:

-   Practical RAG implementation
-   Structured prompt engineering
-   Vector database integration
-   Cost-aware architecture
-   Clean and modular design

It is suitable as a professional portfolio artifact and academic
deliverable.
