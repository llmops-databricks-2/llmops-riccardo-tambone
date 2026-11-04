"""Bounded embedding calls through the course's OpenAI-compatible AI Gateway."""

from openai import OpenAI

from autographrs.config import ProjectConfig
from autographrs.embeddings.models import validate_vector
from autographrs.enrichment.llm import create_gateway_openai_client


class GatewayEmbedder:
    """Use one configured model for document and query embeddings, without truncation."""

    def __init__(
        self,
        config: ProjectConfig,
        client: OpenAI | None = None,
        *,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.config = config
        self.client = client or create_gateway_openai_client()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def embed(self, texts: list[str]) -> tuple[list[list[float]], str]:
        """Validate response indices, dimensions and values; propagate provider failures."""
        if not texts or len(texts) > self.config.embedding_batch_size:
            raise ValueError("Embedding batch must be nonempty and within the configured batch size")
        if any(not text.strip() or len(text) > self.config.embedding_max_chars for text in texts):
            raise ValueError("Embedding text is empty or exceeds embedding_max_chars; no automatic truncation")
        response = self.client.with_options(
            timeout=self.timeout_seconds, max_retries=self.max_retries
        ).embeddings.create(model=self.config.embedding_service, input=texts, encoding_format="float")
        entries = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in entries] != list(range(len(texts))):
            raise ValueError("Gateway response indices do not match the embedding batch")
        vectors = [item.embedding for item in entries]
        for vector in vectors:
            validate_vector(vector, len(vectors[0]))
        if not response.model:
            raise ValueError("Gateway response is missing its model identifier")
        return vectors, response.model
