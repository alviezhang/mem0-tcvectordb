# mem0-tcvectordb

`mem0-tcvectordb` adds Tencent Cloud’s TCVectorDB as a first-class vector store provider for [mem0](https://github.com/mem0ai/mem0). Install it next to `mem0ai` to unlock Tencent Cloud hosting without patching the upstream library.

## Highlights
- Registers a `tcvectordb` provider that works with `Memory`/`AsyncMemory`.
- Re-exports the configuration schema so you get validation and editor hints.
- Keeps close parity with the implementation that ships in the upstream mem0 repo.

## Installation
```bash
pip install mem0-tcvectordb
```

## Usage
```python
import mem0_tcvectordb  # noqa: F401 - registers the provider with mem0
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "tcvectordb",
        "config": {
            "collection_name": "memories",
            "database_name": "mem0",
            "embedding_model_dims": 1536,
            "url": "https://<instance-id>.ap-singapore.tcvdb.tencentcs.com",
            "username": "root",
            "api_key": "tcvdb_api_key",
        }
    }
}

memory = Memory.from_config(config)
memory.add("Alice loves sci-fi movies", user_id="alice")
```

## Configuration Reference
| Parameter | Description | Default |
| --- | --- | --- |
| `collection_name` | Collection where memories are stored | `mem0` |
| `database_name` | Database that owns the collection | `mem0` |
| `embedding_model_dims` | Embedding dimension produced by mem0 | `1536` |
| `url` | TCVectorDB HTTP endpoint | _required_ |
| `username` | Account username | _required_ |
| `api_key` | API key | _required_ |
| `password` | Optional password; defaults to `api_key` | `None` |
| `read_consistency` | `EVENTUAL_CONSISTENCY` or `STRONG_CONSISTENCY` | `EVENTUAL_CONSISTENCY` |
| `shards` | Number of shards | `1` |
| `replicas` | Number of replicas | `1` |
| `metric_type` | Distance metric (`COSINE`, `IP`, `L2`, `HAMMING`) | `COSINE` |
| `index_type` | Index type (e.g., `HNSW`, `FLAT`) | `HNSW` |
| `pool_size` | HTTP connection pool size | `10` |
| `proxies` | Optional proxy configuration | `None` |
| `client` | Pre-built `VectorDBClient` instance | `None` |

mem0 ships embeddings, so TCVectorDB never runs an embedding model. Ensure `embedding_model_dims` matches the mem0 embedding size configured for your deployment.

## Troubleshooting
- `ValueError: TCVectorDB requires url, username, and api_key`: supply credentials in the `vector_store.config` block or pass an authenticated `VectorDBClient`.
- `VectorStoreError: ... failed`: verify the Tencent Cloud endpoint is reachable from your runtime and that `read_consistency`, `metric_type`, and filters use values supported by TCVectorDB.
