from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class TCVectorDBConfig(BaseModel):
    """Configuration for TCVectorDB vector store."""

    collection_name: str = Field("mem0", description="Name of the collection inside TCVectorDB")
    database_name: str = Field("mem0", description="Database name that hosts the collection")
    embedding_model_dims: int = Field(1536, description="Embedding dimension used by mem0")
    client: Optional[Any] = Field(None, description="Existing VectorDBClient instance")
    url: Optional[str] = Field(None, description="HTTP endpoint of TCVectorDB")
    username: Optional[str] = Field(None, description="Account username (typically root)")
    api_key: Optional[str] = Field(None, description="API key for the account")
    password: Optional[str] = Field(None, description="Password, defaults to api_key when omitted")
    timeout: int = Field(10, description="HTTP timeout in seconds")
    read_consistency: str = Field(
        "EVENTUAL_CONSISTENCY",
        description="Read consistency level (EVENTUAL_CONSISTENCY or STRONG_CONSISTENCY)",
    )
    shards: int = Field(1, description="Number of shards for the collection")
    replicas: int = Field(1, description="Replica count for high availability")
    metric_type: str = Field("COSINE", description="Vector distance metric (COSINE, L2, IP, HAMMING)")
    index_type: str = Field("HNSW", description="Vector index type (HNSW, FLAT, IVF_*, etc.)")
    pool_size: int = Field(10, description="HTTP connection pool size")
    proxies: Optional[Dict[str, str]] = Field(None, description="Optional proxy configuration")
    default_indexed_fields: Dict[str, str] = Field(
        default_factory=lambda: {
            "user_id": "string",
            "agent_id": "string",
            "run_id": "string",
            "actor_id": "string",
            "role": "string",
        },
        description="Base payload keys that should always be indexed for filtering with their types",
    )
    indexed_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional payload keys to index on top of the defaults with optional type hints",
    )
    vector_index_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=("Extra parameters for the vector index (e.g. {'M': 36, 'efConstruction': 200} for HNSW)."),
    )

    @model_validator(mode="after")
    def validate_connection(self) -> "TCVectorDBConfig":
        if self.client is None:
            missing = [field for field in ("url", "username", "api_key") if getattr(self, field) in (None, "")]
            if missing:
                raise ValueError(
                    "TCVectorDB requires either an existing client or the following fields: url, username, api_key"
                )
        return self
