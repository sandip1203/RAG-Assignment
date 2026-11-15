from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings


def get_chat_model() -> ChatOpenAI:
    """
    Create an OpenRouter chat model.
    """

    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        temperature=0.2,
    )


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Create an OpenRouter-compatible embedding model.
    """

    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")

    model = settings.embedding_model
    if "/" not in model:
        model = f"openai/{model}"

    return OpenAIEmbeddings(
        model=model,
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        dimensions=settings.embedding_dimension,
        tiktoken_model_name=model.rsplit("/", maxsplit=1)[-1],
    )
