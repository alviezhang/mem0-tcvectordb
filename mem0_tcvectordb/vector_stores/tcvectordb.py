import logging
from functools import wraps
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from pydantic import BaseModel

from mem0.exceptions import VectorStoreError
from mem0.vector_stores.base import VectorStoreBase

try:
    from tcvectordb.client import stub as tc_stub
    from tcvectordb.model.collection import Index
    from tcvectordb.model.enum import ReadConsistency
    from tcvectordb.model.index import (
        FieldType,
        FilterIndex,
        HNSWParams,
        IndexType,
        MetricType,
        VectorIndex,
    )
except ImportError as exc:  # pragma: no cover - dependency is optional
    raise ImportError("TCVectorDB requires extra dependencies. Install with `pip install tcvectordb`.") from exc

logger = logging.getLogger(__name__)

DEFAULT_INDEXED_FIELDS: Dict[str, str] = {
    "user_id": "string",
    "agent_id": "string",
    "run_id": "string",
    "actor_id": "string",
    "role": "string",
}

ALLOWED_FILTER_FIELD_TYPES: Dict[str, FieldType] = {
    "string": FieldType.String,
    "str": FieldType.String,
    "uint64": FieldType.Uint64,
    "array": FieldType.Array,
    "json": FieldType.Json,
}


def _vector_operation(operation: str, passthrough_exceptions: Optional[Sequence[type[Exception]]] = None):
    exceptions = tuple(passthrough_exceptions or ())

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except exceptions:
                raise
            except VectorStoreError:
                raise
            except Exception as exc:  # pragma: no cover - actual SDK errors
                self._raise_vector_store_error(operation, exc)

        return wrapper

    return decorator


class OutputData(BaseModel):
    id: Optional[str]
    payload: Optional[Dict[str, Any]]
    score: Optional[float] = None


class TCVectorDB(VectorStoreBase):
    """TCVectorDB vector store integration."""

    VECTOR_FIELD = "vector"

    INDEX_PARAM_BUILDERS = {"HNSW": "_build_hnsw_params"}

    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        database_name: str,
        client: Optional["tc_stub.VectorDBClient"] = None,
        url: Optional[str] = None,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 10,
        read_consistency: str = "EVENTUAL_CONSISTENCY",
        shards: int = 1,
        replicas: int = 1,
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
        pool_size: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        indexed_fields: Optional[Union[Dict[str, str], List[str]]] = None,
        default_indexed_fields: Optional[Union[Dict[str, str], List[str]]] = None,
        vector_index_params: Optional[Dict[str, Any]] = None,
    ):
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.database_name = database_name
        self.shards = shards
        self.replicas = replicas
        self.metric_type = metric_type
        self.index_type = index_type
        self.timeout = timeout
        self.pool_size = pool_size
        self.proxies = proxies
        self.password = password
        self.indexed_fields = self._normalize_indexed_fields(default_indexed_fields, indexed_fields)
        self.read_consistency = self._parse_read_consistency(read_consistency)
        self.vector_index_params = vector_index_params or {}
        self._unsupported_filter_ops_logged: Set[str] = set()

        if client:
            self.client = client
        else:
            if not url or not username or not api_key:
                raise ValueError("TCVectorDB requires url, username, and api_key when client is not provided.")

            self.client = tc_stub.VectorDBClient(
                url=url,
                username=username,
                key=api_key,
                read_consistency=self.read_consistency,
                timeout=timeout,
                pool_size=pool_size,
                proxies=proxies,
                password=password,
            )

        self.database = None
        self.collection = None
        self.create_col(name=self.collection_name, vector_size=embedding_model_dims, distance=metric_type)

    @_vector_operation("create_collection", passthrough_exceptions=(ValueError,))
    def create_col(
        self,
        name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
    ):
        """Create the database/collection if needed."""
        collection_name = name or self.collection_name
        vector_size = vector_size or self.embedding_model_dims
        metric_type = distance or self.metric_type
        self.database = self.client.create_database_if_not_exists(self.database_name)
        index = self._build_index(vector_size, metric_type)
        self.collection = self.database.create_collection_if_not_exists(
            name=collection_name,
            shard=self.shards,
            replicas=self.replicas,
            index=index,
        )
        self.collection_name = collection_name

    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """Insert vectors as TCVectorDB documents."""
        if not vectors:
            return
        normalized_payloads = self._normalize_payloads(len(vectors), payloads)
        normalized_ids = self._normalize_ids(len(vectors), ids)

        documents = []
        for idx, vector in enumerate(vectors):
            payload = normalized_payloads[idx]
            vector_id = normalized_ids[idx]
            documents.append(self._build_document(vector_id, vector, payload))

        # TCVectorDB supports up to 1000 docs per request; batch accordingly
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                self.collection.upsert(documents=batch)
            except Exception as exc:
                self._raise_vector_store_error("insert", exc)

    @_vector_operation("search", passthrough_exceptions=(ValueError,))
    def search(
        self,
        query: str,
        vectors: Sequence[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[OutputData]:
        """Search for similar vectors."""
        query_vectors = self._normalize_vectors(vectors)
        filter_expression = self._build_filter(filters)
        results = self.collection.search(
            vectors=query_vectors,
            filter=filter_expression,
            limit=limit,
            retrieve_vector=False,
        )
        flattened: List[Dict[str, Any]] = []
        for chunk in results or []:
            if isinstance(chunk, list):
                flattened.extend(chunk)
            else:
                flattened.append(chunk)
        return [self._record_from_document(doc) for doc in flattened]

    @_vector_operation("delete")
    def delete(self, vector_id: str):
        """Delete a vector by ID."""
        self.collection.delete(document_ids=[vector_id])

    @_vector_operation("update")
    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict] = None):
        """Update an existing vector/payload."""
        data = self._build_document(vector_id, vector, payload, include_vector=vector is not None)
        self.collection.update(data=data, document_ids=[vector_id])

    @_vector_operation("get")
    def get(self, vector_id: str) -> Optional[OutputData]:
        """Retrieve a vector by ID."""
        result = self.collection.query(document_ids=[vector_id], limit=1)
        if not result:
            return None
        document = result[0]
        return self._record_from_document(document)

    @_vector_operation("list_collections")
    def list_cols(self):
        """List collections inside the database."""
        collections = self.database.list_collections()
        return [col.collection_name for col in collections]

    @_vector_operation("delete_collection")
    def delete_col(self):
        """Delete the active collection."""
        return self.database.drop_collection(self.collection_name)

    @_vector_operation("describe_collection")
    def col_info(self):
        """Describe collection metadata."""
        collection = self.database.describe_collection(self.collection_name)
        return collection.__dict__

    @_vector_operation("list")
    def list(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = 100) -> List[List[OutputData]]:
        """List memories matching filters."""
        filter_expression = self._build_filter(filters)
        query_limit = limit if limit is not None else None
        documents = self.collection.query(filter=filter_expression, limit=query_limit) or []
        return [[self._record_from_document(doc) for doc in documents]]

    @_vector_operation("reset")
    def reset(self):
        """Reset the collection by truncating it."""
        self.client.truncate_collection(database_name=self.database_name, collection_name=self.collection_name)

    def _build_index(self, vector_size: int, metric_type: str) -> Index:
        index = Index()
        index.add(index=FilterIndex(name="id", field_type=FieldType.String, index_type=IndexType.PRIMARY_KEY))
        for field, field_type in self.indexed_fields.items():
            index.add(
                index=FilterIndex(
                    name=field,
                    field_type=self._field_type(field_type),
                    index_type=IndexType.FILTER,
                )
            )

        metric = self._parse_metric(metric_type)
        params = self._build_vector_index_params()
        vector_index = VectorIndex(
            name=self.VECTOR_FIELD,
            dimension=vector_size,
            index_type=self._parse_index_type(self.index_type),
            metric_type=metric,
            params=params,
        )
        index.add(index=vector_index)
        return index

    def _build_vector_index_params(self):
        if not self.index_type:
            return None

        normalized_type = self.index_type.upper()
        params = self.vector_index_params or {}
        builder_name = self.INDEX_PARAM_BUILDERS.get(normalized_type)
        if builder_name and hasattr(self, builder_name):
            builder = getattr(self, builder_name)
            built_params = builder(params)
            if built_params is not None:
                return built_params

        return params or None

    def _build_hnsw_params(self, params: Dict[str, Any]):
        # HNSW requires both `M` and `efConstruction` and benefits from dimension-based defaults.
        def _get(keys, default=None):
            for key in keys:
                if key in params:
                    return params[key]
            return default

        m = _get(["M", "m"])
        efconstruction = _get(["efConstruction", "efconstruction", "ef_construction"])
        m = m if isinstance(m, (int, float)) else None
        efconstruction = efconstruction if isinstance(efconstruction, (int, float)) else None
        m = int(m) if m is not None else self._recommended_hnsw_m()
        efconstruction = (
            int(efconstruction) if efconstruction is not None else self._recommended_hnsw_efconstruction()
        )
        return HNSWParams(m=m, efconstruction=efconstruction)

    def _recommended_hnsw_m(self) -> int:
        dims = self.embedding_model_dims or 0
        if dims and dims < 512:
            return 16
        return 32

    @staticmethod
    def _recommended_hnsw_efconstruction() -> int:
        return 200

    def _build_document(
        self,
        vector_id: str,
        vector: Optional[Sequence[float]],
        payload: Optional[Dict[str, Any]],
        include_vector: bool = True,
    ) -> Dict[str, Any]:
        document = {"id": str(vector_id)}
        payload_provided = payload is not None
        payload_dict = payload if payload_provided else None

        if include_vector and vector is not None:
            document[self.VECTOR_FIELD] = list(vector)

        if payload_provided and isinstance(payload_dict, dict):
            if "data" in payload_dict:
                document["text"] = payload_dict["data"]

            for field in self.indexed_fields.keys():
                if field in payload_dict:
                    document[field] = payload_dict[field]

            document["payload"] = payload_dict

        return document

    def _normalize_vectors(self, vectors: Sequence[float]) -> List[List[float]]:
        if not vectors:
            raise ValueError("Search vectors cannot be empty")

        first = next(iter(vectors))
        if isinstance(first, (list, tuple)):
            return [list(vec) for vec in vectors]  # type: ignore[arg-type]
        return [list(vectors)]  # type: ignore[list-item]

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        if not filters:
            return None

        return self._build_filter_expression(filters)

    def _build_filter_expression(self, condition: Any) -> Optional[str]:
        if not condition:
            return None

        if isinstance(condition, list):
            expressions = [self._build_filter_expression(item) for item in condition]
            return self._combine_expressions(expressions, operator="and")

        if not isinstance(condition, dict):
            return None

        expressions: List[str] = []
        for key, value in condition.items():
            normalized = key.lower()
            if normalized in {"and", "$and"}:
                sub_expressions = [self._build_filter_expression(item) for item in value or []]
                combined = self._combine_expressions(sub_expressions, operator="and")
                if combined:
                    expressions.append(combined)
            elif normalized in {"or", "$or"}:
                sub_expressions = [self._build_filter_expression(item) for item in value or []]
                combined = self._combine_expressions(sub_expressions, operator="or")
                if combined:
                    expressions.append(combined)
            elif normalized in {"not", "$not"}:
                items = value if isinstance(value, list) else [value]
                sub_expressions = [self._build_filter_expression(item) for item in items]
                combined = self._combine_expressions(sub_expressions, operator="or")
                if combined:
                    grouped = combined if combined.startswith("(") else f"({combined})"
                    expressions.append(f"not {grouped}")
            else:
                field_expression = self._build_field_expression(key, value)
                if field_expression:
                    expressions.append(field_expression)

        return self._combine_expressions(expressions, operator="and")

    def _build_filter_expression_from_dict(self, key: str, value: Dict[str, Any]) -> Optional[str]:
        def _format_sequence(seq: Any) -> str:
            if isinstance(seq, (list, tuple, set)):
                values = seq
            else:
                values = [seq]
            return ", ".join(self._format_value(item) for item in values)

        operator_map = {
            "gte": ">=",
            "gt": ">",
            "lte": "<=",
            "lt": "<",
            "eq": "=",
            "ne": "!=",
        }

        expressions: List[str] = []
        for operator, operand in value.items():
            normalized = operator.lower()
            comparator = operator_map.get(normalized)
            if comparator:
                expressions.append(f"{key} {comparator} {self._format_value(operand)}")
                continue

            if normalized == "in":
                expressions.append(f"{key} in ({_format_sequence(operand)})")
            elif normalized == "nin":
                expressions.append(f"{key} not in ({_format_sequence(operand)})")
            elif normalized in {"include", "includes"}:
                expressions.append(f"{key} include ({_format_sequence(operand)})")
            elif normalized in {"exclude", "not_include", "notinclude"}:
                expressions.append(f"{key} not include ({_format_sequence(operand)})")
            elif normalized in {"include_all", "includeall"}:
                expressions.append(f"{key} include all ({_format_sequence(operand)})")
            else:
                self._log_unsupported_filter_operator(key, operator)

        if expressions:
            return " and ".join(expressions)
        return None

    def _build_field_expression(self, key: str, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str) and value == "*":
            self._log_unsupported_filter_operator(key, "wildcard")
            return None
        if isinstance(value, dict):
            return self._build_filter_expression_from_dict(key, value)
        if isinstance(value, (list, tuple, set)):
            values = ", ".join(self._format_value(v) for v in value)
            return f"{key} in ({values})"
        return f"{key} = {self._format_value(value)}"

    @staticmethod
    def _combine_expressions(expressions: List[Optional[str]], operator: str) -> Optional[str]:
        cleaned = [expr for expr in expressions if expr]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return f" {operator} ".join(f"({expr})" for expr in cleaned)

    def _log_unsupported_filter_operator(self, field: str, operator: str):
        key = f"{field}:{operator}".lower()
        if key in self._unsupported_filter_ops_logged:
            return
        self._unsupported_filter_ops_logged.add(key)
        logger.warning("TCVectorDB filter operator '%s' on field '%s' is not supported and will be ignored.", operator, field)

    def _raise_vector_store_error(self, operation: str, exc: Exception):
        details = {"operation": operation, "collection": self.collection_name, "database": self.database_name}
        suggestion = "Verify TCVectorDB connectivity, credentials, and filter syntax."
        raise VectorStoreError(
            message=f"TCVectorDB {operation} operation failed: {exc}",
            details=details,
            suggestion=suggestion,
        ) from exc

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _parse_metric(metric: str) -> MetricType:
        metric_lower = metric.lower()
        for option in MetricType:
            if option.name.lower() == metric_lower or option.value.lower() == metric_lower:
                return option
        raise ValueError(f"Unsupported metric_type '{metric}' for TCVectorDB")

    @staticmethod
    def _parse_index_type(index_type: str) -> IndexType:
        index_lower = index_type.lower()
        for option in IndexType:
            if option.name.lower() == index_lower or option.value.lower() == index_lower:
                return option
        raise ValueError(f"Unsupported index_type '{index_type}' for TCVectorDB")

    @staticmethod
    def _parse_read_consistency(value: str) -> ReadConsistency:
        try:
            return ReadConsistency[value.upper()]
        except KeyError:
            raise ValueError(f"Unsupported read_consistency '{value}' for TCVectorDB")

    @staticmethod
    def _field_type(value: Union[str, FieldType]) -> FieldType:
        if isinstance(value, FieldType):
            if value not in ALLOWED_FILTER_FIELD_TYPES.values():
                raise ValueError(f"Unsupported TCVectorDB field type '{value.value}'. Allowed types: string, uint64, array, json.")
            return value

        normalized = (value or "string").lower()
        allowed_type = ALLOWED_FILTER_FIELD_TYPES.get(normalized)
        if not allowed_type:
            raise ValueError(
                f"Unsupported TCVectorDB field type '{value}'. Allowed types: string, uint64, array, json."
            )
        return allowed_type

    def _record_from_document(self, document: Dict[str, Any]) -> OutputData:
        payload = document.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        if "text" in document and "data" not in payload:
            payload["data"] = document["text"]

        for field in self.indexed_fields.keys():
            if field in document and field not in payload:
                payload[field] = document[field]

        score = document.get("score")
        if score is None:
            score = document.get("distance")

        return OutputData(id=str(document.get("id")), payload=payload, score=score)

    def _normalize_payloads(
        self,
        expected_length: int,
        payloads: Optional[List[Optional[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        if payloads is None:
            return [{} for _ in range(expected_length)]
        if len(payloads) != expected_length:
            raise ValueError("Length of payloads must match length of vectors.")
        normalized: List[Dict[str, Any]] = []
        for payload in payloads:
            normalized.append(payload or {})
        return normalized

    def _normalize_ids(self, expected_length: int, ids: Optional[List[Any]]) -> List[str]:
        if ids is None:
            return [str(idx) for idx in range(expected_length)]
        if len(ids) != expected_length:
            raise ValueError("Length of ids must match length of vectors.")
        return [str(item) for item in ids]

    @staticmethod
    def _normalize_indexed_fields(
        default_indexed_fields: Optional[Union[Dict[str, str], List[str]]],
        indexed_fields: Optional[Union[Dict[str, str], List[str]]],
    ) -> Dict[str, str]:
        def _pairs(values: Optional[Union[Dict[str, str], List[str]]]) -> List[tuple[str, str]]:
            if not values:
                return []
            if isinstance(values, dict):
                return [(str(field), str(field_type or "string")) for field, field_type in values.items()]
            return [(str(field), "string") for field in values if field]

        normalized: Dict[str, str] = {}
        base_pairs = _pairs(default_indexed_fields) or list(DEFAULT_INDEXED_FIELDS.items())
        for field, field_type in base_pairs + _pairs(indexed_fields):
            if not field:
                continue
            normalized[field] = field_type or "string"
        return normalized
