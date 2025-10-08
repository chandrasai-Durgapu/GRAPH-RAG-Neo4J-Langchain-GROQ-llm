# GRAPH-RAG-Neo4J-Langchain-GROQ-llm
Build GRAPH-RAG using LangChain-GROQ and NEO4J-database which is a graph database

Graph-RAG with Neo4j, LangChain, HuggingFace Embeddings, and GROQ LLM project. It includes all necessary setup, usage, and common troubleshooting issues faced during development.

🚀 Graph-RAG with Neo4j, LangChain, GROQ & HuggingFace Embeddings

This project demonstrates a full Graph-Retrieval-Augmented Generation (Graph-RAG) pipeline using:
---
🧠 LangChain (LLM orchestration)
---
🧵 GROQ (ultra-fast LLM inference)
---
🧱 Neo4j (graph database for knowledge graph & vector store)
---
🔍 HuggingFace Embeddings (sentence-transformers)
---
📚 Wikipedia as knowledge source
---
🌐 Pyvis for interactive graph visualization
---


## 📌 Features

- ✅ Wikipedia data loading and chunking
- ✅ LLM-powered graph extraction (entity + relationship)
- ✅ Storage in Neo4j (nodes, edges, properties)
- ✅ Vector embedding with HuggingFace + Neo4jVector
- ✅ Retrieval from graph and vector index
- ✅ Visualization using Pyvis
- ✅ Structured Cypher query generation
- ✅ LLM response from GROQ via LangChain

---
## 🧰 Technologies Used

| Component | Tech |
|----------|------|
| LLM      | [GROQ](https://console.groq.com/) |
| Framework | [LangChain](https://www.langchain.com/) |
| Graph DB | [Neo4j](https://neo4j.com/) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Loader   | Wikipedia |
| Graph Builder | `LLMGraphTransformer` |
| Vector Store | `Neo4jVector` |
| Visualizer | Pyvis |

---

---

## ⚙️ Setup

### 1. Clone and Install

```bash
git clone https://github.com/chandrasai-Durgapu/GRAPH-RAG-Neo4J-Langchain-GROQ-llm.git
cd GRAPH-RAG-Neo4J-Langchain-GROQ-llm

```
---
## Create Virtual Environment
```bash
python -m venv graph-rag-01
```

## Activate Virtual Environment
```bash
.\graph-rag-01\Scripts\activate

```

## Install Dependencies
```bash
pip install -r requirements.txt
```
---
## Create .env File
```bash
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_api_key
```
---
## Run the Test Application
```bash
python src/main.py
```
---



