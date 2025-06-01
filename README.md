Sequential Queue Chat API

A FastAPI-based chat service that processes requests sequentially, classifies user messages into categories (Irrelevant, Question, Bug, Outage), prioritizes actionable tasks, and provides metrics and semantic search via a FAISS vector index. Uses a local LLM (Qwen/Qwen3-0.6B) to generate bilingual (English/Persian) responses.

Features

Sequential Processing: Ensures only one request is handled at a time using a FIFO queue and background worker.

Message Classification: Tags each message as Irrelevant, Question, Bug, or Outage (multiple tags supported).

Task Prioritization: Extracts Bug and Outage messages into /tasks, prioritizing bug reports.

Metrics: Provides total request count, average response time (ms), and category counts via /metrics.

Semantic Search: Embeds Q&A pairs and supports vector search to return top 10 similar exchanges via /search?q=....

Auto-Chat: Builds conversational context from the last five interactions to generate context-aware replies at /autochat.

Bilingual Support: Automatically responds in Persian or English based on input language.

