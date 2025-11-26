import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tcvectordb.model.index import HNSWParams

from mem0_tcvectordb.vector_stores.tcvectordb import TCVectorDB


def _create_store(**overrides):
    client = MagicMock()
    database = MagicMock()
    collection = MagicMock()

    client.create_database_if_not_exists.return_value = database
    database.create_collection_if_not_exists.return_value = collection
    database.list_collections.return_value = [MagicMock(collection_name="memories")]
    database.describe_collection.return_value = SimpleNamespace(name="memories")

    init_kwargs = dict(
        collection_name="memories",
        embedding_model_dims=1536,
        database_name="mem0",
        client=client,
        metric_type="COSINE",
        index_type="HNSW",
        read_consistency="EVENTUAL_CONSISTENCY",
    )
    init_kwargs.update(overrides)

    store = TCVectorDB(**init_kwargs)

    store.collection = collection
    store.database = database
    return store, client, database, collection


@pytest.fixture
def mock_tcvectordb():
    return _create_store()


def test_initialization_creates_collection(mock_tcvectordb):
    store, _, database, _ = mock_tcvectordb
    database.create_collection_if_not_exists.assert_called_once()
    assert store.collection_name == "memories"


def test_insert_upserts_documents(mock_tcvectordb):
    store, _, _, collection = mock_tcvectordb

    vectors = [[0.1] * 1536]
    payloads = [{"user_id": "alice", "data": "hello"}]
    ids = ["mem_1"]

    store.insert(vectors, payloads=payloads, ids=ids)

    collection.upsert.assert_called_once()
    inserted_docs = collection.upsert.call_args.kwargs["documents"]
    assert inserted_docs[0]["id"] == "mem_1"
    assert inserted_docs[0]["payload"]["data"] == "hello"
    assert inserted_docs[0]["payload"]["user_id"] == "alice"
    assert inserted_docs[0]["user_id"] == "alice"


def test_insert_validates_lengths(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb

    with pytest.raises(ValueError):
        store.insert([[0.1] * 1536], payloads=[], ids=["id1"])

    with pytest.raises(ValueError):
        store.insert([[0.1] * 1536], payloads=[{}], ids=["id1", "id2"])


def test_search_returns_records(mock_tcvectordb):
    store, _, _, collection = mock_tcvectordb

    collection.search.return_value = [
        [
            {"id": "mem_1", "score": 0.9, "payload": {"data": "hello", "user_id": "alice"}},
        ]
    ]

    results = store.search(query="hello", vectors=[0.2] * 1536, limit=2, filters={"user_id": "alice"})

    collection.search.assert_called_once()
    assert results[0].id == "mem_1"
    assert results[0].payload["data"] == "hello"
    assert results[0].score == 0.9


def test_list_returns_records(mock_tcvectordb):
    store, _, _, collection = mock_tcvectordb
    collection.query.return_value = [
        {"id": "mem_2", "payload": {"data": "foo", "user_id": "bob"}},
        {"id": "mem_3", "payload": {"data": "bar", "user_id": "bob"}},
    ]

    results = store.list(filters={"user_id": "bob"}, limit=10)

    collection.query.assert_called_once()
    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0].payload["user_id"] == "bob"


def test_list_default_limit(mock_tcvectordb):
    store, _, _, collection = mock_tcvectordb
    collection.query.return_value = []

    store.list()

    collection.query.assert_called_once()
    assert collection.query.call_args.kwargs["limit"] == 100


def test_build_filter_supports_ranges(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    expression = store._build_filter({"created_at": {"gte": 1, "lte": 2}})
    assert "created_at >=" in expression
    assert "created_at <=" in expression


def test_build_filter_supports_comparison_and_set_ops(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    filters = {
        "score": {"gt": 0.6, "lt": 0.9},
        "hash": {"eq": "abc123"},
        "status": {"ne": "archived"},
        "tags": {"include": ["foo", "bar"]},
        "user_id": {"nin": ["bob", "carol"]},
    }
    expression = store._build_filter(filters)
    assert "(score > 0.6 and score < 0.9)" in expression
    assert '(hash = "abc123")' in expression
    assert '(status != "archived")' in expression
    assert '(tags include ("foo", "bar"))' in expression
    assert '(user_id not in ("bob", "carol"))' in expression


def test_build_filter_supports_logical_operators(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    filters = {
        "AND": [
            {"category": "work"},
            {"priority": {"gte": 7}},
            {
                "OR": [
                    {"status": {"ne": "completed"}},
                    {"status": {"eq": "in_progress"}},
                ]
            },
        ],
        "NOT": [{"tags": {"include": ["archived"]}}],
    }
    expression = store._build_filter(filters)
    assert "(category = \"work\")" in expression
    assert "(priority >= 7)" in expression
    assert "((status != \"completed\") or (status = \"in_progress\"))" in expression
    assert "not (tags include (\"archived\"))" in expression


def test_build_filter_supports_dollar_or_and_not(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    filters = {
        "role": "user",
        "$or": [
            {"agent_id": "bot-1"},
            {"agent_id": "bot-2"},
        ],
        "$not": {"tags": {"include": ["spam"]}},
    }
    expression = store._build_filter(filters)
    assert '(role = "user")' in expression
    assert '((agent_id = "bot-1") or (agent_id = "bot-2"))' in expression
    assert "not (tags include (\"spam\"))" in expression


def test_build_filter_handles_dollar_and(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    filters = {
        "$and": [
            {"role": "user"},
            {"$or": [{"agent_id": "alpha"}, {"agent_id": "beta"}]},
        ]
    }
    expression = store._build_filter(filters)
    assert '(role = "user")' in expression
    assert '((agent_id = "alpha") or (agent_id = "beta"))' in expression


def test_build_filter_returns_none_for_empty_conditions(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    assert store._build_filter({}) is None
    assert store._build_filter({"AND": []}) is None
    assert store._build_filter({"AND": [{"role": None}]}) is None


def test_build_filter_supports_include_all(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    expression = store._build_filter({"tags": {"include_all": ["foo", "bar"]}})
    assert 'tags include all ("foo", "bar")' in expression


def test_build_filter_skips_unsupported_ops(mock_tcvectordb, caplog):
    store, _, _, _ = mock_tcvectordb
    with caplog.at_level(logging.WARNING):
        assert store._build_filter({"tags": {"contains": "foo"}}) is None
        expression = store._build_filter({"category": "*", "role": "user"})
    assert 'role = "user"' in expression
    assert "contains" in caplog.text


def test_custom_indexed_fields_promote_payload_values():
    store, _, _, collection = _create_store(indexed_fields=["team"])

    vectors = [[0.3] * 1536]
    payloads = [{"team": "platform", "data": "note"}]

    store.insert(vectors, payloads=payloads, ids=["id-1"])

    inserted_docs = collection.upsert.call_args.kwargs["documents"]
    assert inserted_docs[0]["team"] == "platform"
    assert inserted_docs[0]["payload"]["team"] == "platform"


def test_record_rehydrates_indexed_fields(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    document = {"id": "1", "payload": {"data": "hello"}, "user_id": "bob"}

    record = store._record_from_document(document)

    assert record.payload["user_id"] == "bob"


def test_update_with_vector_only_does_not_clear_payload(mock_tcvectordb):
    store, _, _, collection = mock_tcvectordb
    new_vector = [0.2] * store.embedding_model_dims

    store.update("mem_1", vector=new_vector, payload=None)

    collection.update.assert_called_once()
    sent_data = collection.update.call_args.kwargs["data"]
    assert TCVectorDB.VECTOR_FIELD in sent_data
    assert "payload" not in sent_data
    assert "id" not in sent_data


def test_update_payload_only_merges_existing_payload(mock_tcvectordb):
    store, _, _, collection = mock_tcvectordb

    store.update("mem_1", vector=None, payload={"user_id": "bob", "tag": "new", "data": "hello"})

    collection.update.assert_called_once()
    update_doc = collection.update.call_args.kwargs["data"]
    assert "vector" not in update_doc
    assert update_doc["payload"]["data"] == "hello"
    assert update_doc["payload"]["user_id"] == "bob"
    assert update_doc["payload"]["tag"] == "new"
    assert update_doc["user_id"] == "bob"
    assert "id" not in update_doc


def test_record_from_document_keeps_zero_scores(mock_tcvectordb):
    store, _, _, _ = mock_tcvectordb
    document = {"id": "1", "payload": {"data": "hello"}, "score": 0.0}

    record = store._record_from_document(document)

    assert record.score == 0.0


def test_additional_indexed_fields_extend_defaults():
    store, _, _, _ = _create_store(indexed_fields={"team": "string"})

    assert "team" in store.indexed_fields
    assert "user_id" in store.indexed_fields  # default retained


def test_default_indexed_fields_override_defaults():
    store, _, _, collection = _create_store(default_indexed_fields={"region": "string"}, indexed_fields={"team": "string"})

    vectors = [[0.5] * 1536]
    payloads = [{"region": "us-west", "team": "platform"}]

    store.insert(vectors, payloads=payloads, ids=["id-2"])

    inserted_docs = collection.upsert.call_args.kwargs["documents"]
    assert inserted_docs[0]["region"] == "us-west"
    assert inserted_docs[0]["team"] == "platform"
    assert list(store.indexed_fields.keys()) == ["region", "team"]


def test_indexed_field_types_configure_filter_types():
    _, _, database, _ = _create_store(indexed_fields={"created_at_ts": "uint64"})
    filter_index = database.create_collection_if_not_exists.call_args.kwargs["index"].indexes["created_at_ts"]
    assert str(filter_index.field_type).lower().endswith("uint64")


def test_indexed_field_invalid_type_raises():
    with pytest.raises(ValueError, match="Allowed types"):
        _create_store(indexed_fields={"created_at_ts": "int64"})


def test_hnsw_params_default_when_missing():
    _, _, database, _ = _create_store(vector_index_params={})

    vector_index = database.create_collection_if_not_exists.call_args.kwargs["index"].indexes[TCVectorDB.VECTOR_FIELD]
    assert isinstance(vector_index.param, HNSWParams)
    assert 4 <= vector_index.param.M <= 64
    assert vector_index.param.efConstruction > 0


def test_hnsw_params_small_dim_uses_recommended_m():
    _, _, database, _ = _create_store(vector_index_params={}, embedding_model_dims=256)

    vector_index = database.create_collection_if_not_exists.call_args.kwargs["index"].indexes[TCVectorDB.VECTOR_FIELD]
    assert isinstance(vector_index.param, HNSWParams)
    assert vector_index.param.M == 16
    assert vector_index.param.efConstruction == 200


def test_hnsw_params_respects_config():
    custom_params = {"M": 40, "efConstruction": 256}
    _, _, database, _ = _create_store(vector_index_params=custom_params)

    vector_index = database.create_collection_if_not_exists.call_args.kwargs["index"].indexes[TCVectorDB.VECTOR_FIELD]
    assert vector_index.param.M == 40
    assert vector_index.param.efConstruction == 256


def test_non_hnsw_index_returns_raw_params():
    _, _, database, _ = _create_store(index_type="IVF_FLAT", vector_index_params={"nlist": 1024})

    vector_index = database.create_collection_if_not_exists.call_args.kwargs["index"].indexes[TCVectorDB.VECTOR_FIELD]
    assert vector_index.param == {"nlist": 1024}
