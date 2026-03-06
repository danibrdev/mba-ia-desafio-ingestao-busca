# Ingestão e Busca Semântica com LangChain e Postgres

Sistema de **Retrieval-Augmented Generation (RAG)** que ingere um PDF e responde perguntas via CLI, com base exclusivamente no conteúdo do documento.

---

## Tecnologias utilizadas

- **Python** + **LangChain**
- **PostgreSQL** + **pgVector**
- **Docker** & **Docker Compose**
- Providers de LLM/Embeddings: **OpenAI** ou **Google Gemini**

---

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose instalados
- Chave de API da **OpenAI** ou do **Google Gemini**

---

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Crie e ative o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas credenciais:

```env
# Provider: 'openai' ou 'gemini'
PROVIDER=gemini

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-5-nano

# Google Gemini
GOOGLE_API_KEY=...
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_LLM_MODEL=gemini-2.5-flash-lite

# PostgreSQL
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=rag_documents

# PDF a ser ingerido
PDF_PATH=document.pdf
```

---

## Execução

### 1. Suba o banco de dados

```bash
docker compose up -d
```

### 2. Ingira o PDF

```bash
python src/ingest.py
```

### 3. Inicie o chat

```bash
python src/chat.py
```

**Exemplo de uso:**

```
Você: Qual o faturamento da Empresa SuperTechIABrazil?

[Assistente]: O faturamento foi de 10 milhões de reais.

---

Você: Quantos clientes temos em 2024?

[Assistente]: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair`, `exit` ou `quit` para encerrar.

---

## Estrutura do projeto

```
├── docker-compose.yml       # PostgreSQL + pgVector
├── requirements.txt         # Dependências
├── .env.example             # Template de variáveis de ambiente
├── document.pdf             # PDF a ser ingerido
├── src/
│   ├── ingest.py            # Script de ingestão do PDF
│   ├── search.py            # Busca semântica + geração de resposta
│   └── chat.py              # Interface CLI
└── tests/
    ├── test_ingest.py
    ├── test_search.py
    └── test_chat.py
```

---

## Testes

```bash
python3 -m pytest tests/ -v
```

---

## Autora

**Danielle Oliveira** — MBA Engenharia de Software com IA · Full Cycle