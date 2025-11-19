"""Helpers to register TCVectorDB components with mem0 at runtime."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict

try:  # Pydantic 2 exposes ModelPrivateAttr for private attribute descriptors.
    from pydantic.fields import ModelPrivateAttr
except Exception:  # pragma: no cover - fallback for unexpected versions.
    ModelPrivateAttr = None  # type: ignore[misc, assignment]


_PACKAGE_ROOT = Path(__file__).resolve().parent
_MODULE_MAP = {
    "mem0.configs.vector_stores.tcvectordb": "configs/vector_stores/tcvectordb.py",
    "mem0.vector_stores.tcvectordb": "vector_stores/tcvectordb.py",
}


def _install_module(module_name: str, relative_path: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    file_path = _PACKAGE_ROOT / relative_path
    loader = SourceFileLoader(module_name, str(file_path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _ensure_vector_store_factory_registration() -> None:
    from mem0.utils.factory import VectorStoreFactory

    provider_map = getattr(VectorStoreFactory, "provider_to_class", {})
    if provider_map.get("tcvectordb") == "mem0.vector_stores.tcvectordb.TCVectorDB":
        return

    updated_map: Dict[str, str] = dict(provider_map)
    updated_map["tcvectordb"] = "mem0.vector_stores.tcvectordb.TCVectorDB"
    VectorStoreFactory.provider_to_class = updated_map


def _ensure_vector_store_config_registration() -> None:
    from mem0.vector_stores.configs import VectorStoreConfig

    provider_attr = getattr(VectorStoreConfig, "_provider_configs", None)
    current_map: Dict[str, str]

    if isinstance(provider_attr, dict):
        current_map = provider_attr
    elif ModelPrivateAttr and isinstance(provider_attr, ModelPrivateAttr):
        current_map = dict(provider_attr.default or {})
    else:
        current_map = {}

    if current_map.get("tcvectordb") == "TCVectorDBConfig":
        return

    current_map["tcvectordb"] = "TCVectorDBConfig"

    if isinstance(provider_attr, dict):
        provider_attr["tcvectordb"] = "TCVectorDBConfig"
    elif ModelPrivateAttr and isinstance(provider_attr, ModelPrivateAttr):
        provider_attr.default = current_map
    else:
        setattr(VectorStoreConfig, "_provider_configs", current_map)


def register_tcvectordb() -> None:
    """Idempotently register the TCVectorDB provider with mem0."""
    for module_name, relative_path in _MODULE_MAP.items():
        _install_module(module_name, relative_path)
    _ensure_vector_store_config_registration()
    _ensure_vector_store_factory_registration()
