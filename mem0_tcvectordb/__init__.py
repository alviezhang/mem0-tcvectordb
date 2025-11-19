"""TCVectorDB plugin for mem0."""

from .patch import register_tcvectordb

# Automatically register when the package is imported so that most users can
# simply `import mem0_tcvectordb` once during startup.
register_tcvectordb()

__all__ = ["register_tcvectordb"]
