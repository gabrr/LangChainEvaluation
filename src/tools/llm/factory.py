from .interface import LLM, ResponseFormat


def llm_factory(
    provider: str = "langchain",
    *,
    model: str,
    system_prompt: str | None = None,
    response_format: ResponseFormat = None,
) -> LLM:
    """Return the selected implementation as an LLM interface.

    Strong model defaults as of September 2026:

        # OpenAI's flagship model
        llm = llm_factory(model="openai:gpt-5.6-sol")

        # Google's strongest stable Flash model
        llm = llm_factory(model="google_genai:gemini-3.8-flash")

        # DeepSeek V4.1 Flash through OpenRouter
        llm = llm_factory(model="openrouter:deepseek/deepseek-v4.1-flash")

        # Gemma 4 26B for Apple silicon
        # Requires langchain-ollama and `ollama pull gemma4:26b-mlx`
        llm = llm_factory(model="ollama:gemma4:26b-mlx")

        response = llm.prompt("Extract the transactions from this statement.")
    """

    if not isinstance(provider, str):
        raise TypeError("provider must be a string.")

    if provider.strip().lower() == "langchain":
        from .langchain import LangChainProvider

        return LangChainProvider(
            model=model,
            system_prompt=system_prompt,
            response_format=response_format,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
