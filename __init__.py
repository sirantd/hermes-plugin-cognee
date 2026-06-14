"""cognee memory provider plugin for Hermes (standalone, user-installed)."""

try:
    from .provider import CogneeMemoryProvider
except ImportError:  # flat import during standalone unit tests
    from provider import CogneeMemoryProvider

__all__ = ["CogneeMemoryProvider", "register"]


def register(ctx) -> None:
    """Hermes plugin entry point — register the cognee memory provider."""
    ctx.register_memory_provider(CogneeMemoryProvider())
