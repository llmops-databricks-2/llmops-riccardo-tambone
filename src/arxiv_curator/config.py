"""Configuration management for Arxiv Curator."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project configuration model."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(..., description="AI Gateway chat model service name")
    embedding_endpoint: str = Field(
        ..., description="Embedding serving endpoint for ai_query, AI Search and model resources"
    )
    embedding_service: str = Field(
        default="course_ops.gateway.embed",
        description="AI Gateway embedding model service name for the OpenAI client",
    )
    bibliography_cache_days: int = Field(
        default=30, ge=0, le=365, description="Successful abstract cache TTL; zero disables"
    )
    embedding_revision: str = Field(default="v1", min_length=1, description="Operator-managed Gateway routing revision")
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_max_chars: int = Field(default=24000, ge=1, description="Safety guard, not a model token limit")
    lakebase_instance: str | None = None
    lakebase_host: str | None = None
    lakebase_database: str = "databricks_postgres"
    lakebase_schema: str | None = None
    lakebase_endpoint: str | None = Field(default=None, description="Optional full projects/.../endpoints/... resource")
    lakebase_user: str | None = Field(default=None, description="OAuth Postgres role; defaults to current SDK identity")
    search_timeout_seconds: int = Field(default=1800, ge=1, le=7200)
    usage_policy_id: str | None = None
    experiment_path: str = Field(..., description="MLflow experiment path for this environment")
    warehouse_id: str = Field(..., description="Warehouse ID")
    genie_space_id: str | None = Field(None, description="Genie space ID for MCP integration")
    system_prompt: str = Field(
        default="You are a helpful AI assistant that helps users find and understand research papers.",
        description="System prompt for the agent",
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance

        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        values = {
            "usage_policy_id": config_data.get("usage_policy_id"),
            **config_data[env],
        }
        return cls(**values)

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"


def load_config(config_path: str = "project_config.yml", env: str = "dev") -> ProjectConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        ProjectConfig instance

    """
    # Handle relative paths from notebooks
    if not Path(config_path).is_absolute():
        # Try to find config in parent directories
        current = Path.cwd()
        for _ in range(3):  # Search up to 3 levels
            candidate = current / config_path
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent

    return ProjectConfig.from_yaml(config_path, env)


def get_env() -> str:
    """Get current environment from environment variable.

    Returns:
        Environment name (dev, acc, or prd)

    """
    return os.getenv("ENV", "dev")
