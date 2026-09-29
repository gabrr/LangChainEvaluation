# LangChain Evaluation Lab

Small lab for exploring agent and graph-workflow evaluations with LangChain and LangSmith.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Provider credentials configured from `.env.example`

## Setup

```bash
cp .env.example .env
uv sync
```

Add the credentials for the providers you want to use to `.env`.

## Run

```bash
uv run --env-file .env langgraph dev
```
