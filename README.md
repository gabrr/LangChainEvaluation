# LangChain Evaluation Lab

Small lab for exploring agent evaluations with LangSmith.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- A root `.env` with `OPENROUTER_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT`

## Run

```bash
uv sync
uv run --env-file .env langgraph dev
```
