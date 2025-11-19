"""End-to-end tests that hit a live TCVectorDB instance."""

from __future__ import annotations

import os
import time
from contextlib import suppress
from typing import Callable, Dict, Iterable, List, Optional
from uuid import uuid4

import pytest

from mem0_tcvectordb.vector_stores.tcvectordb import TCVectorDB
from tests.conftest import require_env

pytestmark = pytest.mark.integration

_TRUTHY = {"1", "true", "yes", "on"}


def test_tcvectordb_round_trip(integration_env):
    """Verify inserts/searches/list operations against the real service."""
    store = _build_store()

    if _is_truthy(os.environ.get("MEM0_TEST_RESET_COLLECTION", "0")):
        store.reset()

    doc_id = f"tcvectordb-int-{uuid4().hex}"
    payload = {
        "user_id": integration_env["test_user_id"],
        "hash": uuid4().hex[:12],
        "created_at_ts": int(time.time()),
        "data": f"Integration test payload for {integration_env['test_user_id']}",
        "tags": ["integration", "tcvectordb"],
    }

    vector = _sample_vector(store.embedding_model_dims)

    try:
        store.insert([vector], payloads=[payload], ids=[doc_id])

        record = _wait_for(lambda: store.get(doc_id), condition=lambda result: result is not None)
        assert record is not None
        assert record.id == doc_id
        assert record.payload["user_id"] == payload["user_id"]

        search_results = _wait_for(
            lambda: store.search(
                query=integration_env["search_query"],
                vectors=vector,
                limit=3,
                filters={"user_id": payload["user_id"], "hash": payload["hash"]},
            ),
            condition=lambda results: _contains_id(results, doc_id),
        )
        assert _contains_id(search_results, doc_id)

        listed_results = _wait_for(
            lambda: store.list(filters={"user_id": payload["user_id"]}, limit=20),
            condition=lambda batches: _contains_id_in_groups(batches, doc_id),
        )
        assert _contains_id_in_groups(listed_results, doc_id)
    finally:
        with suppress(Exception):
            store.delete(doc_id)


def _contains_id(results: Optional[Iterable], expected_id: str) -> bool:
    if not results:
        return False
    return any(getattr(item, "id", None) == expected_id for item in results)


def _contains_id_in_groups(groups: Optional[Iterable[List]], expected_id: str) -> bool:
    if not groups:
        return False
    return any(_contains_id(group, expected_id) for group in groups)


def _wait_for(
    fetcher: Callable[[], Optional[Iterable]],
    *,
    condition: Callable[[Optional[Iterable]], bool],
    timeout: float = 30,
    interval: float = 1.0,
):
    deadline = time.time() + timeout
    last_value = None
    while time.time() < deadline:
        value = fetcher()
        last_value = value
        if condition(value):
            return value
        time.sleep(interval)
    pytest.fail(f"Timed out waiting for TCVectorDB to reflect the change. Last value: {last_value}")


def _build_store() -> TCVectorDB:
    indexed_fields = _parse_indexed_fields(os.environ.get("TCVECTORDB_INDEXED_FIELDS"))
    default_indexed_fields = _parse_indexed_fields(os.environ.get("TCVECTORDB_DEFAULT_INDEXED_FIELDS"))

    kwargs = {
        "collection_name": os.environ.get("TCVECTORDB_COLLECTION", "mem0"),
        "embedding_model_dims": _int_from_env("TCVECTORDB_EMBED_DIM", fallback="1536"),
        "database_name": os.environ.get("TCVECTORDB_DATABASE", "mem0"),
        "url": require_env("TCVECTORDB_URL"),
        "username": require_env("TCVECTORDB_USERNAME"),
        "api_key": require_env("TCVECTORDB_API_KEY"),
        "password": os.environ.get("TCVECTORDB_PASSWORD") or None,
        "read_consistency": os.environ.get("TCVECTORDB_CONSISTENCY", "EVENTUAL_CONSISTENCY"),
        "metric_type": os.environ.get("TCVECTORDB_METRIC", "COSINE"),
        "index_type": os.environ.get("TCVECTORDB_INDEX", "HNSW"),
    }

    maybe_params = _parse_index_params(os.environ.get("TCVECTORDB_INDEX_PARAMS"))
    if maybe_params:
        kwargs["vector_index_params"] = maybe_params
    if indexed_fields:
        kwargs["indexed_fields"] = indexed_fields
    if default_indexed_fields:
        kwargs["default_indexed_fields"] = default_indexed_fields
    return TCVectorDB(**kwargs)


def _sample_vector(dims: int) -> List[float]:
    return [(idx % 17) / 17.0 for idx in range(dims)]


def _parse_indexed_fields(raw: Optional[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    if not raw:
        return fields
    for chunk in raw.split(","):
        cleaned = chunk.strip()
        if not cleaned:
            continue
        if ":" in cleaned:
            field, hint = cleaned.split(":", 1)
        else:
            field, hint = cleaned, "string"
        name = field.strip()
        field_type = hint.strip() or "string"
        if name:
            fields[name] = field_type
    return fields


def _parse_index_params(raw: Optional[str]) -> Dict[str, int]:
    if not raw:
        return {}
    params: Dict[str, int] = {}
    for chunk in raw.split(","):
        cleaned = chunk.strip()
        if not cleaned or "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            params[key] = int(value)
        except ValueError:
            continue
    return params


def _int_from_env(name: str, *, fallback: str) -> int:
    raw_value = os.environ.get(name, fallback)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        raise RuntimeError(f"Environment variable '{name}' must be an integer.")


def _is_truthy(value: Optional[str]) -> bool:
    return bool(value and value.strip().lower() in _TRUTHY)
